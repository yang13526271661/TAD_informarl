# https://blog.csdn.net/weixin_39925939/article/details/121360884

import torch
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor, matmul

class BigCatConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add') 

    def forward(self, x, edge_index):
        x = x
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        print('message')
        print('x_j:', x_j)
        return x_j
    
    def message_and_aggregate(self, x, adj_t):
        print('message_and_aggregate')
        return matmul(adj_t, x, reduce=self.aggr)

# 定义图的特征和边
x = torch.eye(4)
edge_index = torch.tensor([[1,2,3,3,0,0,0,1], [0,0,0,1,1,2,3,3]])
model = BigCatConv()
out = model(x, edge_index)
print('x:', x)