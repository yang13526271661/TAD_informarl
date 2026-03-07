import sys

file_path = "/data/yangxiaodi_space/TAD-informarl/InforMARL-main/multiagent/TAD_rand_2t1a1d.py"
with open(file_path, "r") as f:
    text = f.read()

# I want to insert the graph methods right before target_policy
target_idx = text.find("def target_policy(")
if target_idx == -1:
    print("Could not find target_policy")
    sys.exit(1)

# Backtrack to find the comment
comment_idx = text.rfind("'''\nlow-level policy for TADs", 0, target_idx)
if comment_idx == -1:
    print("Could not find comment")
    sys.exit(1)

insert_idx = comment_idx

graph_methods = """
    def update_graph(self, world):
        \"\"\"
        Construct a graph from the cached distances.
        Nodes are entities in the environment
        Edges are constructed by thresholding distances
        \"\"\"
        dists = world.cached_dist_mag
        # just connect the ones which are within connection
        # distance and do not connect to itself
        connect = np.array((dists <= self.max_edge_dist) * (dists > 0)).astype(int)
        from scipy import sparse
        sparse_connect = sparse.csr_matrix(connect)
        sparse_connect = sparse_connect.tocoo()
        row, col = sparse_connect.row, sparse_connect.col
        edge_list = np.stack([row, col])
        world.edge_list = edge_list
        if world.graph_feat_type == "global":
            world.edge_weight = dists[row, col]
        elif world.graph_feat_type == "relative":
            world.edge_weight = dists[row, col]

    def _get_entity_feat_global(self, entity, world):
        \"\"\"
        Returns: ([velocity, position, size, entity_type])
        \"\"\"
        if self.use_CL:
            self.set_CL(glv.get_value('CL_ratio'))

        pos = entity.state.p_pos
        vel = entity.state.p_vel
        Radius = entity.size
        
        if "attacker" in entity.name:
            entity_type = entity_mapping["attacker"]
        elif "target" in entity.name:
            entity_type = entity_mapping["target"]
        elif "defender" in entity.name:
            entity_type = entity_mapping["defender"]
        else:
            raise ValueError(f"{entity.name} not supported")

        import numpy as np
        return np.hstack([pos, vel, Radius, entity_type])

    def _get_entity_feat_relative(self, agent, entity, world):
        \"\"\"
        Returns: ([relative_velocity, relative_position, size, entity_type])
        \"\"\"
        if self.use_CL:
            self.set_CL(glv.get_value('CL_ratio'))

        pos = entity.state.p_pos - agent.state.p_pos
        vel = entity.state.p_vel - agent.state.p_vel
        Radius = entity.size
        
        if "attacker" in entity.name:
            entity_type = entity_mapping["attacker"]
        elif "target" in entity.name:
            entity_type = entity_mapping["target"]
        elif "defender" in entity.name:
            entity_type = entity_mapping["defender"]
        else:
            raise ValueError(f"{entity.name} not supported")

        import numpy as np
        return np.hstack([pos, vel, Radius, entity_type])

    def get_id(self, agent):
        import numpy as np
        return np.array([agent.global_id])

    def graph_observation(self, agent, world):
        num_entities = len(world.entities)
        # node observations
        node_obs = []
        if world.graph_feat_type == "global":
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_global(entity, world)
                node_obs.append(node_obs_i)
        elif world.graph_feat_type == "relative":
            for i, entity in enumerate(world.entities):
                node_obs_i = self._get_entity_feat_relative(agent, entity, world)
                node_obs.append(node_obs_i)
        else:
            raise ValueError(f"Graph Feature Type {world.graph_feat_type} not supported")

        import numpy as np
        node_obs = np.array(node_obs)
        adj = world.cached_dist_mag

        return node_obs, adj

"""

new_text = text[:insert_idx] + graph_methods + text[insert_idx:]

with open(file_path, "w") as f:
    f.write(new_text)

print("Patched successfully")
