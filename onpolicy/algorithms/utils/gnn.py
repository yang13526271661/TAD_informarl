import numpy as np
from scipy import sparse
import torch
from torch import Tensor
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, to_dense_batch
from torch_geometric.nn import TransformerConv


import argparse
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
import torch.jit as jit
from .util import init, get_clones
import torch.nn.functional as F


class EmbedConv(MessagePassing):
    """
    实体嵌入与初始特征融合层
    """
    def __init__(self, 
                input_dim:int, 
                num_embeddings:int, 
                embedding_size:int, 
                hidden_size:int,
                layer_N:int,
                use_orthogonal:bool,
                use_ReLU:bool,
                use_layerNorm:bool,
                add_self_loop:bool,
                edge_dim:int=0):
        # [核心修复 1]：将 aggr='add' 改为 aggr='mean'！
        # 防止在密集编队或扎堆包夹时，节点特征因为简单相加而导致数值爆炸和梯度消失
        super(EmbedConv, self).__init__(aggr='mean')
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layerNorm else nn.Identity()
        self.init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)

        self.lin1 = nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)
        
        self.layers = nn.ModuleList()
        for _ in range(layer_N):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(self.active_func)
            self.layers.append(self.layer_norm)
        
        self._initialize_weights()

    def _initialize_weights(self):
        gain = nn.init.calculate_gain('relu' if isinstance(self.active_func, nn.ReLU) else 'tanh')
        self.init_method(self.lin1.weight, gain=gain)
        nn.init.constant_(self.lin1.bias, 0)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.init_method(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x:Union[Tensor, OptPairTensor], edge_index:Adj, edge_attr:OptTensor=None):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j:Tensor, edge_attr:OptTensor):
        node_feat_j = x_j[:, :-1]
        entity_type_j = x_j[:, -1].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        x = self.lin1(node_feat)
        x = self.active_func(x)
        x = self.layer_norm(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x


class SimplifiedAttentionNet(nn.Module):
    """
    实现 Communication Enhanced Network (CEN)
    具备真正的多层堆叠、非线性激活与残差连接机制。
    """
    def __init__(self,
                input_dim: int,
                num_embeddings: int,
                embedding_size: int,
                hidden_size: int,
                layer_N: int,
                use_ReLU: bool,
                graph_aggr: str,
                global_aggr_type: str,
                embed_hidden_size: int,
                embed_layer_N: int,
                embed_use_orthogonal: bool,
                embed_use_ReLU: bool,
                embed_use_layerNorm: bool,
                embed_add_self_loop: bool,
                max_edge_dist: float,
                edge_dim: int = 1,
                num_heads: int = 3,
                concat_heads: bool = False): 
        super(SimplifiedAttentionNet, self).__init__()
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.edge_dim = edge_dim
        self.max_edge_dist = max_edge_dist
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type

        # 1. 实体嵌入与初始特征融合
        self.embed_layer = EmbedConv(
            input_dim=input_dim - 1,  
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=embed_hidden_size,
            layer_N=embed_layer_N,
            use_orthogonal=embed_use_orthogonal,
            use_ReLU=embed_use_ReLU,
            use_layerNorm=embed_use_layerNorm,
            add_self_loop=embed_add_self_loop,
            edge_dim=edge_dim,
        )

        # 2. [核心修复 2]：动态构建多层 GNN，恢复多跳通信与深层思考能力
        self.attn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        in_channels = embed_hidden_size
        for _ in range(layer_N):
            self.attn_layers.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=hidden_size,
                    heads=num_heads,
                    concat=concat_heads,  
                    dropout=0.0,
                    beta=True      
                )
            )
            # 动态计算下一层的输入维度
            out_channels = hidden_size * (num_heads if concat_heads else 1)
            self.norms.append(nn.LayerNorm(out_channels))
            in_channels = out_channels

    def forward(self, batch: Data, agent_id: Tensor):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        
        # 步骤 1：融合节点特征、类型嵌入和边特征
        x = self.embed_layer(x, edge_index, edge_attr)

        # 步骤 2：[核心修复 3] 包含残差连接和激活函数的深层消息传递
        for conv, norm in zip(self.attn_layers, self.norms):
            x_new = conv(x, edge_index)
            x_new = self.active_func(x_new)  # 非线性激活
            x_new = norm(x_new)              # 层归一化防止梯度消失
            
            # 残差连接 (Residual Connection): 极大缓解图神经过平滑 (Over-smoothing)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new

        # 步骤 3：极速维度还原 
        batch_size = agent_id.shape[0]
        num_nodes = x.shape[0] // batch_size
        x = x.view(batch_size, num_nodes, -1)

        if self.graph_aggr == "node":
            return self.gatherNodeFeats(x, agent_id)
        elif self.graph_aggr == "global":
            return self.graphAggr(x)

        raise ValueError(f"Invalid graph_aggr: {self.graph_aggr}")

    @staticmethod
    def process_adj(adj: Tensor, max_edge_dist: float) -> Tuple[Tensor, Tensor]:
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)

        connect_mask = (adj > 0).float()
        adj = adj * connect_mask
        
        if adj.dim() == 3:
            batch_size, num_nodes, _ = adj.shape
            edge_index = adj.nonzero(as_tuple=False)
            edge_attr = adj[edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]]
            batch = edge_index[:, 0] * num_nodes
            edge_index = torch.stack([batch + edge_index[:, 1], batch + edge_index[:, 2]], dim=0)
        else:
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            edge_attr = adj[edge_index[0], edge_index[1]]

        edge_attr = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr
        return edge_index, edge_attr

    def gatherNodeFeats(self, x: Tensor, idx: Tensor):
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()
        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1)
            idx_tmp = idx_tmp.repeat(1, num_feats).unsqueeze(1)
            gathered_node = x.gather(1, idx_tmp).squeeze(1)
            out.append(gathered_node)
        return torch.cat(out, dim=1)

    def graphAggr(self, x: Tensor):
        if self.global_aggr_type == "mean":
            return x.mean(dim=1)
        elif self.global_aggr_type == "max":
            return x.max(dim=1)[0]
        elif self.global_aggr_type == "add":
            return x.sum(dim=1)
        else:
            raise ValueError(f"Invalid global_aggr_type: {self.global_aggr_type}")

class GNNBase(nn.Module):
    """
        A Wrapper for constructing the Base graph neural network.
        This uses TransformerConv from Pytorch Geometric
        https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.TransformerConv
        and embedding layers for entity types
    """
    def __init__(self, args:argparse.Namespace, 
                node_obs_shape:Union[List, Tuple],
                edge_dim:int, graph_aggr:str):
        super(GNNBase, self).__init__()

        self.args = args
        self.hidden_size = args.gnn_hidden_size
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads

        self.gnn = SimplifiedAttentionNet(input_dim=node_obs_shape, edge_dim=edge_dim,
                    num_embeddings=args.num_embeddings,
                    embedding_size=args.embedding_size,
                    hidden_size=args.gnn_hidden_size,
                    layer_N=args.gnn_layer_N,
                    use_ReLU=args.gnn_use_ReLU,
                    graph_aggr=graph_aggr,
                    global_aggr_type=args.global_aggr_type,
                    embed_hidden_size=args.embed_hidden_size,
                    embed_layer_N=args.embed_layer_N,
                    embed_use_orthogonal=args.use_orthogonal,
                    embed_use_ReLU=args.embed_use_ReLU,
                    embed_use_layerNorm=args.use_feature_normalization,
                    embed_add_self_loop=args.embed_add_self_loop,
                    max_edge_dist=args.max_edge_dist,
                    num_heads=args.gnn_num_heads,
                    concat_heads=args.gnn_concat_heads)
        
        self.out_dim = args.gnn_hidden_size * (args.gnn_num_heads if args.gnn_concat_heads else 1)
        
    def forward(self, node_obs:Tensor, adj:Tensor, agent_id:Tensor):
        batch_size, num_nodes, _ = node_obs.shape
        edge_index, edge_attr = SimplifiedAttentionNet.process_adj(adj, self.gnn.max_edge_dist)
        
        x = node_obs.view(-1, node_obs.size(-1))
        batch = torch.arange(batch_size, device=node_obs.device).repeat_interleave(num_nodes)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
        x = self.gnn(data, agent_id)

        if self.gnn.graph_aggr == "node":
            return x.view(batch_size, -1)
        return x