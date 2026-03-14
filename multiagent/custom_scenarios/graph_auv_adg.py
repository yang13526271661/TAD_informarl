import numpy as np
import os  
import torch
from multiagent.TAD_core import World, Target, Attacker, Defender, Landmark
from multiagent.scenario import BaseScenario
from scipy import sparse
from multiagent.TAD_util import map_attacker_action, map_defender_action

torch.set_num_threads(1) # <--- 强制封杀底层多线程争抢
# --- 论文真实场景参数设定 (放大 2.5 倍以适应 500 步训练) ---
R_SAFE = 5.0     
R_C = 50.0       
R_B = 100.0      
V_A_MAX = 5.4    # (2.16 * 2.5) 攻击者最大速度 
V_D_MAX = 3.0    # (1.2 * 2.5) 防御者最大速度

# ========================================== #
# Defender 的内置"陪练"策略 (比例导引制导 PNG)
# 作用: 作为一个聪明的对手, 逼迫 MAPPO-Attacker 学习更强策略
# ========================================== #
def defender_policy(agent, world):
    if len(world.attackers) == 0 or agent.done: 
        return np.array([0., 0.])
    
    target = world.targets[0]
    agent_in_rc = np.linalg.norm(agent.state.p_pos - target.state.p_pos) < R_C
    
    # 筛选出在感知范围内的攻击者
    observable_attackers = []
    for a in world.attackers:
        if a.done: continue
        a_in_rc = np.linalg.norm(a.state.p_pos - target.state.p_pos) < R_C
        dist = np.linalg.norm(a.state.p_pos - agent.state.p_pos)
        
        # 严格遵守论文：同在 Rc 内，或在 25米 内
        if a_in_rc or dist <= 25.0: 
            observable_attackers.append(a)
            
    # 如果雷达里没有任何攻击者，原地待命/巡逻
    if not observable_attackers:
        return np.array([0., 0.])
    
    # 在可感知的敌人中，找到最近的攻击者作为拦截目标
    attacker = min(observable_attackers, key=lambda a: np.linalg.norm(a.state.p_pos - agent.state.p_pos))
    cl_ratio = getattr(world, 'CL_ratio', 1.0)

    # 恢复真正的导弹杀手本色！
    N_m = 1.0 + 3.0 * cl_ratio
    V_m, V_d = attacker.state.p_vel, agent.state.p_vel
    x_md = agent.state.p_pos - attacker.state.p_pos
    dist_norm = np.linalg.norm(x_md)
    
    if dist_norm < 1e-4: 
        return np.array([0., 0.])
    
    e_md = x_md / dist_norm
    closing_vel = -(V_d - V_m) 
    
    # ================= 核心修复：主发动机推力 + 侧向修正力 =================
    # 1. 基础追踪推力：永远指向攻击者 (防守者指向攻击者的向量是 -e_md)
    base_thrust = -e_md * agent.max_accel
    
    if np.linalg.norm(closing_vel) > 0:
        los_rate = np.cross(x_md, closing_vel) / (dist_norm ** 2)
        
        # 2. PNG 侧向修正力：用于提前量拦截
        accel_mag = N_m * np.linalg.norm(closing_vel) * los_rate
        e_ad = np.array([-e_md[1], e_md[0]]) 
        png_thrust = e_ad * accel_mag
        
        # 3. 物理合力 = 追踪推力 + PNG 侧向修正
        total_thrust = base_thrust + png_thrust
        
        # [新增避免队友相撞逻辑] 防守者之间的避撞斥力
        for other_d in world.defenders:
            if other_d is agent: continue
            vec_from_other = agent.state.p_pos - other_d.state.p_pos
            dist_other = np.linalg.norm(vec_from_other)
            edge_dist = dist_other - agent.size - other_d.size
            if edge_dist < 10.0:
                edge_dist_safe = max(edge_dist, 0.1)
                safe_dist_other = max(dist_other, 1e-4)
                repel_mag = ((10.0 - edge_dist_safe) / 10.0) * agent.max_accel * 1.5
                total_thrust += (vec_from_other / safe_dist_other) * repel_mag
        
        # 限制总加速度不超过物理极限
        norm_thrust = np.linalg.norm(total_thrust)
        if norm_thrust > agent.max_accel:
            return (total_thrust / norm_thrust) * agent.max_accel
        return total_thrust
        
    return base_thrust
    # ======================================================================

# ========================================== #
# Attacker 的内置"陪练"策略 (人工势场 APF)
# 作用: 作为防守者预训练阶段的对手，提供基础的突防能力
# ========================================== #
def attacker_policy(agent, world):
    if getattr(agent, 'done', False):
        return np.array([0., 0.])
        
    target = world.targets[0]
    
    # 提取课程进度 (0.0 到 1.0)
    cl_ratio = getattr(world, 'CL_ratio', 1.0)
    train_mode = getattr(world, 'train_mode', 'attacker')
    
    # 1. 目标引力方向 (f_v)
    vec_to_target = target.state.p_pos - agent.state.p_pos
    dist_to_target = np.linalg.norm(vec_to_target)
    f_v = (vec_to_target / dist_to_target) if dist_to_target > 0 else np.zeros(2)
    
    # 2. 防守者斥力方向 (f_d)
    f_d = np.zeros(2)
    for defender in world.defenders:
        vec_from_def = agent.state.p_pos - defender.state.p_pos
        center_dist = np.linalg.norm(vec_from_def)
        edge_dist = center_dist - agent.size - defender.size 
        
        # 训练防御者时，给防御者一个 25m 的“威慑光环”
        SAFE_EDGE_DIST = 25.0 if train_mode == 'defender' else 10.0
        
        if edge_dist < SAFE_EDGE_DIST:
            edge_dist_safe = max(edge_dist, 0.1) 
            safe_center_dist = max(center_dist, 1e-4)
            
            # ================= 核心修改：平滑分段梯度斥力 =================
            if train_mode == 'defender':
                if edge_dist_safe >= 20.0:
                    # 【20~25m】极弱排斥 (0 ~ 0.2)：此时引力(0.8)依然主导
                    # 攻击者感觉“似乎有危险，但还能冲”，会继续向基地逼近并稍微偏转
                    repulsion_mag = 0.2 * ((25.0 - edge_dist_safe) / 5.0)
                elif edge_dist_safe >= 15.0:
                    # 【15~20m】中等排斥 (0.2 ~ 0.8)：引力与斥力开始抗衡
                    # 攻击者开始被明显向侧面推开，尝试进行大角度绕路切入
                    repulsion_mag = 0.2 + 0.6 * ((20.0 - edge_dist_safe) / 5.0)
                elif edge_dist_safe >= 10.0:
                    # 【10~15m】强烈排斥 (0.8 ~ 2.8)：斥力压倒引力
                    # 攻击者感到极度危险，放弃冲锋，准备调头撤退
                    repulsion_mag = 0.8 + 2.0 * ((15.0 - edge_dist_safe) / 5.0)
                else:
                    # 【0~10m】致命排斥 (2.8 ~ 7.8)：绝对的物理禁区
                    # 疯狂逃命模式，确保防守者贴脸时依然能把攻击者逼退
                    repulsion_mag = 2.8 + 5.0 * ((10.0 - edge_dist_safe) / 10.0)
            else:
                # 训练攻击者（阶段 A）时，维持原来的单调线性逻辑
                repulsion_mag = ((SAFE_EDGE_DIST - edge_dist_safe) / SAFE_EDGE_DIST) * 10.0
            # ==========================================================
            
            f_d += (vec_from_def / safe_center_dist) * repulsion_mag
    # 3. 涡旋场方向 (f_t)
    f_t = np.array([-f_v[1], f_v[0]]) 
    
    # ================= 核心修改：区分训练模式的 APF 权重 =================
    if train_mode == 'defender':
        # 【阶段 B：训练防守者】
        # 此时攻击者是陪练，必须具有较强的“畏惧感”。
        # 当防守者靠近时，极高的斥力权重 w_d 会迫使攻击者调头逃跑，防守者由此学会“驱逐”
        w_t = 0.8  # 依然想去基地
        w_d = 2.0  # 极强的退让斥力（防守者靠近时，保命优先于进基地）
        w_v = 0.2  # 稍微加一点侧切，防止陷入死胡同
        
        # 训练防御者时，攻击者的推力直接满血，不需要随课程慢慢解锁
        curr_thrust_ratio = 1.0 
    else:
        # 【阶段 A：训练攻击者】
        # 保持之前的动态难度：一开始无视防守者死冲，后期慢慢学会躲避
        w_t = 0.8 - 0.3 * cl_ratio  
        w_d = 0.0 + 0.8 * cl_ratio  
        w_v = 0.0 + 0.6 * cl_ratio  
        curr_thrust_ratio = 0.4 + 0.35 * cl_ratio
    # =======================================================================
    
    noise = np.random.normal(0, 0.1, 2)
    combined_force = w_v * f_t + w_d * f_d + w_t * f_v + noise
    
    norm_force = np.linalg.norm(combined_force)
    if norm_force > 0:
        return (combined_force / norm_force) * (agent.max_accel * curr_thrust_ratio)
    return np.zeros(2)

def target_policy(agent, world):
    return np.array([0., 0.])

# ------------------------------------------

class Scenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()
        # 严格对齐论文 Table II 的参数
        self.sensing_radius_A = 25.0  # R_A^sen
        self.comm_radius_A = 25.0     # R_A^com
        self.sensing_radius_D = 25.0  # R_D^sen
        self.comm_radius_D = 25.0     # R_D^com
        self.catch_radius = R_SAFE

    def make_world(self, args):
        world = World()
        # 课程初始难度：默认从 0 起步，后续由 env.set_CL 写入 CL_ratio
        world.CL_ratio = 0.0
        # 1. 动态读取训练模式和轮次
        # --- 新增：把碰撞变成有弹性的碰碰车，而不是致命的台球 ---
        world.contact_force = 100.0  # 极大降低排斥刚度，避免因为力太大导致加速度爆炸 (原来是20)
        world.damping = 0.5         # 增加阻尼，让碰撞更像是减速带而不是反弹 (原来是0.5)
        
        world.train_mode = os.environ.get('TRAIN_MODE', 'attacker')
        world.current_iter = int(os.environ.get('CURRENT_ITER', '1'))
        world.world_length = 500
        self.num_target = getattr(args, 'num_target', 1)
        self.num_attacker = getattr(args, 'num_attacker', 1)
        self.num_defender = getattr(args, 'num_defender', 2)
        world.agents = []
        world.landmarks = []
        
        # --- 还原地图要素 ---
        zone_b = Landmark()
        zone_b.name = 'outer_zone'; zone_b.collide = False; zone_b.movable = False; zone_b.size = R_B; zone_b.color = np.array([0.7, 0.85, 0.95]) 
        world.landmarks.append(zone_b)

        zone_c = Landmark()
        zone_c.name = 'defense_zone'; zone_c.collide = False; zone_c.movable = False; zone_c.size = R_C; zone_c.color = np.array([0.95, 0.8, 0.8]) 
        world.landmarks.append(zone_c)

        for i in range(self.num_target):
            target = Target()
            target.name = 'target %d' % i; target.collide = False; target.movable = False; target.size = R_SAFE; target.color = np.array([0.45, 0.95, 0.45])
            target.action_callback = target_policy
            world.agents.append(target)

        # ================= 还原论文 Alg. 1 策略池采样机制 ================= #
        self.strategy_pool = []
        if world.train_mode == 'defender' and world.current_iter > 1:
            # 防守者先训练，遍历加载所有历史轮次的攻击者模型
            for iter_idx in range(1, world.current_iter):
                path = f"../onpolicy/results/GraphMPE/graph_auv_adg/rmappo/SelfPlay_Iter{iter_idx}_Attacker/models/actor_structure.pt"
                if os.path.exists(path):
                    self.strategy_pool.append(path)
        elif world.train_mode == 'attacker' and world.current_iter > 1:
            # 攻击者后训练，可以加载一直到【当前轮次】刚训练好的防守者模型
            for iter_idx in range(1, world.current_iter + 1):
                path = f"../onpolicy/results/GraphMPE/graph_auv_adg/rmappo/SelfPlay_Iter{iter_idx}_Defender/models/actor_structure.pt"
                if os.path.exists(path):
                    self.strategy_pool.append(path)
                    
        # 预先加载最新模型作为默认项
        world.frozen_actor = None
        if self.strategy_pool:
            try:
                world.frozen_actor = torch.load(self.strategy_pool[-1], map_location=torch.device('cpu'))
                world.frozen_actor.eval()
            except:
                pass
        # ================= 核心修改：动态加载上一轮的完整模型 ================= #
        frozen_model_path = None
        if world.train_mode == 'defender' and world.current_iter > 1:
            # 防守者训练，只能加载上一轮的攻击者模型 (因为当前轮的还没训练)
            frozen_model_path = f"../onpolicy/results/GraphMPE/graph_auv_adg/rmappo/SelfPlay_Iter{world.current_iter-1}_Attacker/models/actor_structure.pt"
        elif world.train_mode == 'attacker' and world.current_iter > 1:
            # 攻击者训练，加载当前轮刚训练好的防守者模型
            frozen_model_path = f"../onpolicy/results/GraphMPE/graph_auv_adg/rmappo/SelfPlay_Iter{world.current_iter}_Defender/models/actor_structure.pt"
        
        world.frozen_actor = None
        if frozen_model_path and os.path.exists(frozen_model_path):
            try:
                # 直接通过完整模型文件加载
                world.frozen_actor = torch.load(frozen_model_path, map_location=torch.device('cpu'))
                
                # ================= 核心洗脑包：彻底抹除所有子模块的 CUDA 记忆 ================= #
                def _force_cpu(model):
                    model.to(torch.device('cpu'))
                    for m in model.modules(): # 遍历网络里所有的子组件 (Actor, GNN, RNN...)
                        if hasattr(m, 'tpdv'):
                            m.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
                        if hasattr(m, 'device'):
                            m.device = torch.device('cpu')
                
                _force_cpu(world.frozen_actor)
                # ============================================================================== #

                world.frozen_actor.eval()
                print(f"成功加载陪练模型: {frozen_model_path}")
            except Exception as e:
                print(f"模型加载失败: {e}")

        for i in range(self.num_attacker):
            attacker = Attacker()
            attacker.name = 'attacker %d' % i; attacker.collide = True; attacker.movable = True; attacker.max_speed = V_A_MAX; attacker.max_accel = 5.0; attacker.size = 3.0; attacker.color = np.array([0.95, 0.2, 0.2])
            # 如果当前是防守者在训练，那么攻击者就是陪练，绑定加载的模型策略
            if world.train_mode == 'defender':
                attacker.action_callback = self.opponent_policy
            world.agents.append(attacker)
            
        for i in range(self.num_defender):
            defender = Defender()
            defender.name = 'defender %d' % i; defender.collide = True; defender.movable = True; defender.max_speed = V_D_MAX; defender.max_accel = 3.0; defender.size = 3.0; defender.color = np.array([0.2, 0.2, 0.95])
            # 如果当前是攻击者在训练，防守者是陪练，绑定加载的模型策略
            if world.train_mode == 'attacker':
                defender.action_callback = self.opponent_policy
            world.agents.append(defender)

        world.targets = [a for a in world.agents if isinstance(a, Target)]
        world.attackers = [a for a in world.agents if isinstance(a, Attacker)]
        world.defenders = [a for a in world.agents if isinstance(a, Defender)]
        
        world.cache_dists = True  
        world.graph_mode = True
        self.reset_world(world)
        return world
    
    def opponent_policy(self, agent, world):
        # 1. 陪练如果已经阵亡或出界，强制禁止其动作
        if getattr(agent, 'done', False):
            return np.array([0., 0.])

        # 2. 预训练边界：第 1 轮 (Iter 1) 强制双方使用规则 (PNG/APF) 搏斗
        if getattr(world, 'current_iter', 1) == 1:
            if isinstance(agent, Defender):
                return defender_policy(agent, world)  
            elif isinstance(agent, Attacker):
                return attacker_policy(agent, world)  
            else:
                return np.array([0., 0.])

        # 3. 安全兜底：如果意外没读取到模型，降级为规则
        if getattr(world, 'frozen_actor', None) is None:
            if isinstance(agent, Defender):
                return defender_policy(agent, world)  
            else:
                return attacker_policy(agent, world)
                
        # ================= 下面是你之前不小心删掉的神经网络推理代码 =================
        # 4. 提取观测
        obs = self.observation(agent, world)
        ag_id = self.get_id(agent, world)
        node_obs, adj = self.graph_observation(agent, world)
        
        if not hasattr(agent, 'rnn_states'):
            agent.rnn_states = torch.zeros(1, 1, 64)
        masks = torch.ones(1, 1)

        # 5. 网络前向传播，计算出 action 变量！
        with torch.inference_mode():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            ag_id_t = torch.as_tensor(ag_id, dtype=torch.float32).unsqueeze(0)
            node_obs_t = torch.as_tensor(node_obs, dtype=torch.float32).unsqueeze(0)
            adj_t = torch.as_tensor(adj, dtype=torch.float32).unsqueeze(0)

            # 这里输出了 action！
            action, action_log_probs, new_rnn_states = world.frozen_actor(
                obs_t, node_obs_t, adj_t, ag_id_t, agent.rnn_states, masks
            )
            agent.rnn_states = new_rnn_states
        # =======================================================================
            
        raw_action = action.squeeze(0).cpu().numpy()
        
        # 6. 调用全局统一映射函数 (彻底解决不一致问题)
        if isinstance(agent, Attacker):
            return map_attacker_action(agent, world, raw_action)
        else:
            return map_defender_action(agent, raw_action)
        

    def reset_world(self, world):
        # ================= 终极修复 1：彻底清空“前世”的物理缓存 ================= #
        world.cached_dist_mag = None
        world.cached_dist_vec = None

        # 在 graph_auv_adg.py 的 reset_world 中
        # 在每个 Episode 开始前，按概率从策略池中抽一个对手模型
        eta_p = 0.35 # 论文 Table II 设定的概率

        # 【修复】：判断当前是否为评估/渲染模式，如果是，强制 100% 使用最新策略！
        is_eval = os.environ.get('CURRENT_ITER') is not None and "Render" in os.environ.get('EXPERIMENT_NAME', '')

        if hasattr(self, 'strategy_pool') and len(self.strategy_pool) > 0:
            if (not is_eval) and np.random.rand() > eta_p and len(self.strategy_pool) > 1:
                # 仅在训练时：随机抽取历史老策略
                chosen_model_path = np.random.choice(self.strategy_pool[:-1])
            else:
                # 渲染时 或 随机到35%概率时：使用最新策略
                chosen_model_path = self.strategy_pool[-1]

            world.frozen_actor = torch.load(chosen_model_path, map_location=torch.device('cpu'))
            
            # ================= 核心洗脑包：彻底抹除所有子模块的 CUDA 记忆 ================= #
            def _force_cpu(model):
                model.to(torch.device('cpu'))
                for m in model.modules():
                    if hasattr(m, 'tpdv'):
                        m.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
                    if hasattr(m, 'device'):
                        m.device = torch.device('cpu')
            
            _force_cpu(world.frozen_actor)
            # ============================================================================== #
            
            world.frozen_actor.eval()
        # 强制所有人（包括 Target）洗白状态，满血复活
        for agent in world.agents:
            agent.done = False  
            agent.state.p_pos = np.zeros(world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.action.u = np.zeros(world.dim_p) # 卸掉上一局的残留惯性力
            
            agent.rnn_states = torch.zeros(1, 1, 64)
            # [核心修复 1]：彻底清空所有跨回合记忆字典，防止 Step 0 梯度爆炸
            if hasattr(agent, 'prev_dist_for_def'):
                agent.prev_dist_for_def = {}
            if hasattr(agent, 'prev_dist_for_neff'):
                agent.prev_dist_for_neff = {}
            if hasattr(agent, 'rewarded_defenders'):
                agent.rewarded_defenders = set()
            if hasattr(agent, 'prev_dist_to_att'):
                agent.prev_dist_to_att = None
        # ========================================================================= #

        cl_ratio = getattr(world, 'CL_ratio', 1.0) 
        
        for landmark in world.landmarks:
            landmark.state.p_pos = np.array([0., 0.])
            landmark.state.p_vel = np.array([0., 0.])

        world.targets[0].state.p_pos = np.array([0., 0.])
        world.targets[0].state.p_vel = np.array([0., 0.])
        
        # ================= 按照新思路重构的非对称课程学习动力学 ================= #
        # 1. 动态目标半径 (保留原有的逻辑，训练初期放宽判定)
        current_safe_radius = 10.0 - (10.0 - 5.0) * cl_ratio
        world.targets[0].size = current_safe_radius
        self.catch_radius = current_safe_radius

        if world.train_mode == 'attacker':
            # 【阶段 A：训练攻击者】
            # 攻击者（主角）：始终保持最高机动性，方便其冲锋
            current_v_a_max = 5.4
            current_accel_a_max = 5.0
            
            # 防守者（陪练）：初始极速 0.5，推力 0.5 (像乌龟一样极低) -> 随着轮数增加，恢复到满血 3.0
            current_v_d_max = 0.5 + (3.0 - 0.5) * cl_ratio
            current_accel_d_max = 0.5 + (3.0 - 0.5) * cl_ratio
            
        else:
            # 【阶段 B：训练防守者】
            # 防守者（主角）：始终保持正常的机动性水平
            current_v_d_max = 3.0
            current_accel_d_max = 3.0
            
            # 攻击者（陪练）：初始极速 3.5 (比防守者的 3.0 高一点点)，推力 3.5 -> 随轮数增加，飙升到极速 5.4 和推力 5.0
            current_v_a_max = 3.5 + (5.4 - 3.5) * cl_ratio
            current_accel_a_max = 3.5 + (5.0 - 3.5) * cl_ratio
        # ================================================================= #
        
        min_radius = R_C + 5.0
        current_spawn_radius = min_radius + (R_B - min_radius) * cl_ratio
        
        # 攻击者出生逻辑
        for i, attacker in enumerate(world.attackers):
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(50.0, 100.0) 
            attacker.state.p_pos = np.array([
                current_spawn_radius * np.cos(angle),
                current_spawn_radius * np.sin(angle)
            ])
            attacker.state.p_vel = np.zeros(world.dim_p)
            
            # 【重要：确保动态速度和加速度被正确赋值】
            attacker.max_speed = current_v_a_max
            attacker.max_accel = current_accel_a_max
            attacker.done = False
            attacker.prev_dist = np.linalg.norm(attacker.state.p_pos - world.targets[0].state.p_pos)

        curr_R_D_max = 25.0 + (R_C - 25.0) * cl_ratio
        
        # 防御者出生逻辑
        for i, defender in enumerate(world.defenders):
            angle = np.random.uniform(0, 2*np.pi)
            r_def = np.random.uniform(R_SAFE, curr_R_D_max)
            defender.state.p_pos = np.array([np.cos(angle)*r_def, np.sin(angle)*r_def])
            defender.state.p_vel = np.zeros(world.dim_p)
            
            # 【重要：确保动态速度和加速度被正确赋值】
            defender.max_speed = current_v_d_max
            defender.max_accel = current_accel_d_max
            defender.done = False

    

    def reward(self, agent, world):
        if isinstance(agent, Attacker): 
            return self.attacker_reward(agent, world)
        elif isinstance(agent, Defender):
            return self.defender_reward(agent, world)
        return 0.0

    def attacker_reward(self, agent, world):
        if getattr(agent, 'done', False):
            return 0.0
            
        rew = 0.0
        target = world.targets[0]
        
        # =======================================================
        # 1. 终端价值函数 (Terminal Value Function, 论文 Eq. 14)
        # =======================================================
        if any(np.linalg.norm(a.state.p_pos - target.state.p_pos) <= self.catch_radius for a in world.attackers):
            agent.done = True
            world.targets[0].done = True
            return 50.0  # 胜利时不直接 return，因为还需要结算之前的卡位和距离奖励！
            
        if np.linalg.norm(agent.state.p_pos - target.state.p_pos) >= R_B:
            agent.done = True
            return -50.0 # 彻底失败（出界/死亡）必须直接 Return！否则它死后的废弃坐标会扰乱下面的阿波罗尼斯计算产生巨额负分！

        if world.current_step >= world.world_length - 1:
            agent.done = True
            return -51.0  # 让你转圈！时间到了照样扣 50！
        
        # =======================================================
        # 2. 瞬时奖励项 (Instantaneous Reward, 论文 Eq. 13)
        # =======================================================
        # (a) 时间步惩罚 (hstep)
        rew -= 0.02

        # ================= 动态物理半径碰撞判定 (攻击者视角) =================
        W_O = 20.0           
        
        g_o_att = 0.0
        is_hard_collision_att = False
        
        for other_agent in world.defenders + world.attackers:
            if other_agent == agent: 
                continue  
                
            dist = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
            
            # 【终极修复】：动态获取真实物理半径
            R_BODY = agent.size + other_agent.size  
            R_SAFE_COLLISION = R_BODY + 3.0       # 3.0
            
            if dist <= (R_BODY + 0.0): # 硬碰撞判定，允许有 1 米的误差容忍
                is_hard_collision_att = True
                break  
            elif dist < R_SAFE_COLLISION:
                g_o_att += (R_SAFE_COLLISION - dist) / (R_SAFE_COLLISION - R_BODY)
                
        if is_hard_collision_att:
            # 【修复神风特攻】碰撞致死的惩罚必须 >= 出界惩罚！否则它会为了逃避出界的 -50.0 而故意撞死（Reward Hacking）
            rew -= W_O
            agent.done = True # 解除注释！攻击者被撞毁必须立刻阵亡，停止计算！
        else:
            rew -= W_O * g_o_att * 0.2
        # =================================================================================

        # ================= 破局补丁：微弱的进攻引导 =================
        curr_dist = np.linalg.norm(agent.state.p_pos - target.state.p_pos)
        if not hasattr(agent, 'prev_dist'):
            agent.prev_dist = curr_dist
            
        # 【修改 2】：动态靠近奖励。圈外给小甜头，敢进圈就给大甜头！
        if curr_dist < R_C:
            # 进了雷达圈，风险极高，重赏勇夫 (权重提高到 0.2)
            approach_reward = (agent.prev_dist - curr_dist) * 0.2
        else:
            # 圈外安全区，微小引导 (权重保持 0.05)
            approach_reward = (agent.prev_dist - curr_dist) * 0.05
        rew += approach_reward
        agent.prev_dist = curr_dist
        # ==========================================================

        # (c) 阿波罗尼斯威慑惩罚 (-w_theta * g_theta) 严格还原论文 Eq.4 - Eq.6
        w_theta = 0.5 # 1.0, 0.125也可以
        g_theta = 0.0
        
        # 1. 射线参数初始化
        R_A_sen = self.sensing_radius_A
        num_rays = 31 # 射线数量，决定检测分辨率
        delta_theta = np.pi / (num_rays - 1)
        
        vec_at = target.state.p_pos - agent.state.p_pos
        dist_at = np.linalg.norm(vec_at)
        
        if dist_at > 0:
            theta_target = np.arctan2(vec_at[1], vec_at[0])
            # 攻击者的有效探索扇区 Omega_j (目标方向正负 90 度)
            angles = np.linspace(theta_target - np.pi/2, theta_target + np.pi/2, num_rays)
            
            d_theta_vec = np.zeros(2)
            N_hit = 0
            
            actual_v_d = world.defenders[0].max_speed if len(world.defenders) > 0 else 3.0
            actual_v_a = agent.max_speed
            
            lam = actual_v_d / actual_v_a   # 放心除！物理源头已保证分母永远大于分子
            lam_sq = lam ** 2
            
            # 2. 发射射线进行阿波罗尼斯圈相交检测 (Ray-casting)
            for angle in angles:
                u_vec = np.array([np.cos(angle), np.sin(angle)])
                min_hit_dist = R_A_sen + 1.0
                
                for def_agent in world.defenders:
                    # 【核心修复】：攻击者视角，变量名是 agent，且感知严格限制在 25m
                    dist_def_att = np.linalg.norm(def_agent.state.p_pos - agent.state.p_pos)
                    if dist_def_att > 25.0:
                        continue
                        
                    vec_ad = def_agent.state.p_pos - agent.state.p_pos
                    
                    dist_ad = np.linalg.norm(vec_ad)
                    
                    if dist_ad < 1e-4:
                        continue
                        
                    # 计算防守者的阿波罗尼斯圆参数 O 和 R_o
                    center_rel = vec_ad / (1.0 - lam_sq) 
                    R_O = (lam / (1.0 - lam_sq)) * dist_ad
                    
                    # 射线与圆的交点计算 (几何投影法)
                    proj_len = np.dot(center_rel, u_vec)
                    if proj_len > 0: # 圆心在射线前方
                        dist_line_sq = np.dot(center_rel, center_rel) - proj_len**2
                        R_O_sq = R_O**2
                        
                        if dist_line_sq <= R_O_sq: # 射线穿过了威慑圆
                            # 计算最近的表面交点距离
                            intersect_dist = proj_len - np.sqrt(R_O_sq - dist_line_sq)
                            if 0 < intersect_dist < min_hit_dist:
                                min_hit_dist = intersect_dist
                
                # 3. 记录命中并累加威慑阻碍向量 (还原论文 Eq. 4)
                if min_hit_dist <= R_A_sen:
                    N_hit += 1
                    weight = (R_A_sen - min_hit_dist) / R_A_sen
                    d_theta_vec += weight * u_vec
            
            # 4. 计算有效攻击率 A_ratio 和最终惩罚项 g_theta (还原论文 Eq. 5 & 6)
            A_ratio = 1.0 - (N_hit / num_rays)
            
            # 依据 Eq. 4 归一化 d_theta 向量
            coef = delta_theta / np.pi
            d_theta_vec *= coef
            
            # 这里的 norm_d_theta_sq 就是论文公式中的 ||d_j^theta||_2^2
            norm_d_theta_sq = np.dot(d_theta_vec, d_theta_vec)
            
            if norm_d_theta_sq > 1e-8:
                # 【核心修复 3】：分母加上 1e-8，彻底粉碎除零导致的 NaN！
                denominator = np.sqrt(norm_d_theta_sq) * dist_at + 1e-8
                cos_omega = np.clip(np.dot(d_theta_vec, vec_at) / denominator, -1.0, 1.0)
                
                omega = np.arccos(cos_omega)
                
                # 仅当夹角小于 pi/3 时，威慑生效并乘以平方极值
                if omega < np.pi / 3.0:
                    g_theta = norm_d_theta_sq * (np.pi / 3.0 - omega)
                    
        rew -= w_theta * g_theta
        # print("Attacker Reward Debug: hstep_penalty=-0.02, collision_penalty={:.2f}, g_theta_penalty={:.4f}, total_rew={:.4f}".format( W_O if is_hard_collision_att else W_O * g_o_att * 0.05, w_theta * g_theta, rew))
        return rew

    def defender_reward(self, agent, world):
        if agent.done: 
            return 0.0
            
        rew = 0.0
        target = world.targets[0]
        
        # =======================================================
        # 1. 终端价值函数 (Terminal Value Function)
        # =======================================================
        # 失败判定：任何活着的攻击者进入了目标安全区
        if any(np.linalg.norm(a.state.p_pos - target.state.p_pos) <= (self.catch_radius + a.size) for a in world.attackers):
            agent.done = True
            world.targets[0].done = True
            return -50.0  # 基地被毁，防守者接受 -50 分的终极惩罚！
        # ================= 核心修改：动态物理半径碰撞判定 =================
        W_O = 10.0
        
        g_o_def = 0.0
        is_hard_collision_def = False
        
        # --- 双层碰撞判定 (同时规避其他防守者和攻击者) ---
        for other_agent in world.defenders:
            if other_agent == agent: 
                continue 

            dist_ad = np.linalg.norm(agent.state.p_pos - other_agent.state.p_pos)
            
            # 【终极修复】：动态获取双方真实的物理半径之和 (3.0 + 3.0 = 6.0)
            R_BODY = agent.size + other_agent.size  
            # 安全预警区必须大于物理半径！(比如设置在身体外围 2 米处，即 10.0)
            R_SAFE_COLLISION = R_BODY + 4.0         
            
            if dist_ad <= (R_BODY + 0.0): # 硬碰撞判定，允许有 1 米的误差容忍
                is_hard_collision_def = True
    
            elif dist_ad < R_SAFE_COLLISION:
                g_o_def += (R_SAFE_COLLISION - dist_ad) / (R_SAFE_COLLISION - R_BODY)
                
        if is_hard_collision_def:
            rew -= W_O   # 恢复硬碰撞惩罚(-10)，必须让他们感到疼，不敢无脑扎堆抢人头
            # agent.done = True # 防守者被撞毁必须立刻阵亡，停止计算！
        else:
            rew -= W_O * g_o_def * 0.1 # 恢复软避障的排斥场梯度，让他们靠近时就自动散开

        # ================= 破局补丁：稳定的密集的防守引导 (追击最近的攻击者) =================
        if len(world.attackers) > 0:
            nearest_attacker = min(world.attackers, key=lambda a: np.linalg.norm(agent.state.p_pos - a.state.p_pos))
            curr_dist_to_att = np.linalg.norm(agent.state.p_pos - nearest_attacker.state.p_pos)
            if not hasattr(agent, 'prev_dist_to_att') or agent.prev_dist_to_att is None:
                agent.prev_dist_to_att = curr_dist_to_att
            
            # 只有“攻击者-目标”距离进入 R_C 内，才给追击引导奖励
            dist_att_to_target = np.linalg.norm(nearest_attacker.state.p_pos - target.state.p_pos)
            if dist_att_to_target <= R_C:
                approach_reward_def = (agent.prev_dist_to_att - curr_dist_to_att) * 0.0
                # rew += approach_reward_def
            # 无论是否在 R_C 内，都更新记忆，避免下一次突变
            agent.prev_dist_to_att = curr_dist_to_att
        # =======================================================================

        # ================= 核心修改 2：驱逐判定逻辑 =================
        g_d = 0.0
        # 【修复2】：清理了重复复制的循环，逻辑现在非常干净
        for attacker in world.attackers:
            if not hasattr(attacker, 'rewarded_defenders'):
                attacker.rewarded_defenders = set()

            dist_t = np.linalg.norm(attacker.state.p_pos - target.state.p_pos)
            is_out = (dist_t >= R_B)
            
            # 只有将敌人驱逐出边界，才算有效拦截
            if is_out:
                attacker.done = True  
                if agent.name not in attacker.rewarded_defenders:
                    if np.linalg.norm(agent.state.p_pos - attacker.state.p_pos) <= self.sensing_radius_D:
                        g_d += 1.0  
                        attacker.rewarded_defenders.add(agent.name)
                        
        # 胜利判定：所有攻击者都已经出界
        if all(a.done for a in world.attackers) and (not world.targets[0].done):
            agent.done = True
            rew += 50.0
           

        # =======================================================
        # 2. 划分 D_eff (受控区) 和 D_neff (无人区) 
        # =======================================================
        D_eff = []
        D_neff = []
        global_D_eff = set()
        
        # --- 重新定义感知逻辑 (符合论文 IoUT 网络设定) ---
        agent_in_rc = np.linalg.norm(agent.state.p_pos - target.state.p_pos) < R_C

        for d in world.defenders:
            d_in_rc = np.linalg.norm(d.state.p_pos - target.state.p_pos) < R_C
            for a in world.attackers:
                if a.done: continue
                a_in_rc = np.linalg.norm(a.state.p_pos - target.state.p_pos) < R_C
                dist_da = np.linalg.norm(d.state.p_pos - a.state.p_pos)
                # 【修复4】：去掉 d_in_rc 的限制！只要攻击者在圈内，就算受控！
                if a_in_rc or dist_da <= self.sensing_radius_D:
                    global_D_eff.add(a.name)

        for attacker in world.attackers:
            if attacker.done: continue
            a_in_rc = np.linalg.norm(attacker.state.p_pos - target.state.p_pos) < R_C
            dist_to_me = np.linalg.norm(agent.state.p_pos - attacker.state.p_pos)
            # 【修复5】：去掉 agent_in_rc 的限制！
            if a_in_rc or dist_to_me <= self.sensing_radius_D:
                D_eff.append(attacker)
            if attacker.name not in global_D_eff:
                D_neff.append(attacker)

        # =======================================================
        # 3. 瞬时奖励计算 
        # =======================================================
        w_theta, w_s, w_d = 5.0, 0.1, 50.0 #20.0
        w_f, w_e, w_step = 0.1, 0.1, -0.02  # 1.0, 0.8, -0.02
        
        g_s, g_f, g_e, g_theta = 0.0, 0.0, 0.0, 0.0  

        # ------------------ (a) 计算 g_s (被你误删的推移奖励，必须补回！) ------------------
        for attacker in D_eff:
            curr_dist = np.linalg.norm(attacker.state.p_pos - target.state.p_pos)
            if not hasattr(attacker, 'prev_dist_for_def'):
                attacker.prev_dist_for_def = {}
            prev_dist = attacker.prev_dist_for_def.get(agent.name, curr_dist)
            
            g_s += (curr_dist - prev_dist) 
            # print(f'g_s: {g_s}')
            attacker.prev_dist_for_def[agent.name] = curr_dist

        # ------------------ (b) 计算 g_f 和 g_e ------------------
        for attacker in D_neff:
            curr_dist = np.linalg.norm(attacker.state.p_pos - target.state.p_pos)
            if not hasattr(attacker, 'prev_dist_for_neff'):
                attacker.prev_dist_for_neff = {}
            prev_dist = attacker.prev_dist_for_neff.get(agent.name, curr_dist)
            if curr_dist < R_C:
                g_f += (curr_dist - prev_dist)
            # print(f'g_f: {g_f}')
            attacker.prev_dist_for_neff[agent.name] = curr_dist

        active_attackers = [a for a in world.attackers if not a.done]
        if len(active_attackers) > 0:
            intrusion_sum = 0.0
            for a in active_attackers:
                dist_t = np.linalg.norm(a.state.p_pos - target.state.p_pos)
                # 【核心修复】：攻击者边缘触及 R_C 即算入侵
                if dist_t < R_C + a.size:  
                    intrusion_sum += (R_C + a.size - dist_t) / (R_C + a.size)
            g_e = -intrusion_sum / len(active_attackers)
            # print(f'g_e: {g_e}')

        # ------------------ (c) 计算 g_theta (纯净的 31 根射线法) ------------------
        R_A_sen = self.sensing_radius_A
        num_rays = 31 
        delta_theta = np.pi / (num_rays - 1)

        for attacker in D_eff:
            vec_at = target.state.p_pos - attacker.state.p_pos
            dist_at = np.linalg.norm(vec_at)
            
            if dist_at > 0:
                attacker_in_rc = np.linalg.norm(attacker.state.p_pos - target.state.p_pos) < R_C
                
                theta_target = np.arctan2(vec_at[1], vec_at[0])
                angles = np.linspace(theta_target - np.pi/2, theta_target + np.pi/2, num_rays)
                
                d_theta_vec = np.zeros(2)
                N_hit = 0
                
                actual_v_d = agent.max_speed
                actual_v_a = attacker.max_speed
                
                lam = actual_v_d / actual_v_a   # 放心除！
                lam_sq = lam ** 2
                
                for angle in angles:
                    u_vec = np.array([np.cos(angle), np.sin(angle)])
                    min_hit_dist = R_A_sen + 1.0
                    
                    # 【核心修改】：把 world.defenders 改成了 [agent]
                    # 逻辑：系统现在只检测“我（当前防御者）”一个人是否遮挡了射向基地的射线
                    for def_agent in [agent]: 
                        def_agent_in_rc = np.linalg.norm(def_agent.state.p_pos - target.state.p_pos) < R_C
                        dist_def_att = np.linalg.norm(def_agent.state.p_pos - attacker.state.p_pos)
                        is_visible = attacker_in_rc or (dist_def_att <= self.sensing_radius_D)
                        
                        if not is_visible:
                            continue
                        vec_ad = def_agent.state.p_pos - attacker.state.p_pos
                        dist_ad = np.linalg.norm(vec_ad)
                        
                        if dist_ad < 1e-4: continue
                            
                        center_rel = vec_ad / (1.0 - lam_sq) 
                        R_O = (lam / (1.0 - lam_sq)) * dist_ad
                        
                        proj_len = np.dot(center_rel, u_vec)
                        if proj_len > 0:
                            dist_line_sq = np.dot(center_rel, center_rel) - proj_len**2
                            R_O_sq = R_O**2
                            
                            if dist_line_sq <= R_O_sq:
                                intersect_dist = proj_len - np.sqrt(R_O_sq - dist_line_sq)
                                if 0 < intersect_dist < min_hit_dist:
                                    min_hit_dist = intersect_dist
                    
                    if min_hit_dist <= R_A_sen:
                        N_hit += 1
                        weight = (R_A_sen - min_hit_dist) / R_A_sen
                        d_theta_vec += weight * u_vec
                
                A_ratio = 1.0 - (N_hit / num_rays)
                coef = delta_theta / np.pi
                d_theta_vec *= coef
                
                norm_d_theta_sq = np.dot(d_theta_vec, d_theta_vec)
                
                if norm_d_theta_sq > 1e-8:
                    # 【核心修复 3】：分母加上 1e-8，彻底粉碎除零导致的 NaN！
                    denominator = np.sqrt(norm_d_theta_sq) * dist_at + 1e-8
                    cos_omega = np.clip(np.dot(d_theta_vec, vec_at) / denominator, -1.0, 1.0)
                    
                    omega = np.arccos(cos_omega)
                    
                    if omega < np.pi / 3.0:
                        g_theta += norm_d_theta_sq * (np.pi / 3.0 - omega)
        # print(f'g_theta: {g_theta}')   
        # print(f"w_step: {w_step}")    
        # ------------------ 奖励加总 (绝对不能再加别的东西了) ------------------
        # 将这句放在 return rew 的正上方
        
        # print(f"[{agent.name}] g_theta: {g_theta:.4f} | w_step: {w_step:.4f} | g_s: {g_s:.4f} | g_e: {g_e:.4f}")
        rew += (w_theta * g_theta) + (w_s * g_s) + (w_d * g_d) + (w_f * g_f) + (w_e * g_e) + w_step 
        # 在返回 rew 之前加入以下逻辑 (大约在 510 行左右)
        dist_to_target = np.linalg.norm(agent.state.p_pos - target.state.p_pos)

        # 领地约束：如果防御者离开 R_C (75m) 太远，给予线性惩罚
        if dist_to_target > R_C:
        # 这里的 0.01 是惩罚强度，可以根据实验微调
            rew -= 0.02 * (dist_to_target - R_C) 
    
        # 离岗惩罚：严禁离开最大边界 R_B (125m)
        if dist_to_target > R_B:
            rew -= 2.0  # 施加一个较大的瞬时惩罚，迫使其调头
        return rew
    
    def observation(self, agent, world):
        # 基础向量观测必须使用相对目标的相对坐标，并进行【归一化】
        target = world.targets[0]
        
        # 除以地图最大半径 R_B (100.0)，将其压缩到 [-1, 1] 左右
        rel_pos_norm = (agent.state.p_pos - target.state.p_pos) / R_B
        
        # 除以最大速度极限 (假设两者相向而行最大约 4.0)，压缩到 [-1, 1]
        rel_vel_norm = agent.state.p_vel / 10.0 
        
        return np.concatenate([rel_pos_norm, rel_vel_norm])
    
    def graph_observation(self, agent, world):
        all_entities = world.targets + world.attackers + world.defenders
        num_nodes = len(all_entities)
        node_obs = []
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        target = world.targets[0]
        # 判断每个实体是否在核心防御区 (IoUT 基站覆盖范围)
        in_rc = np.array([np.linalg.norm(e.state.p_pos - target.state.p_pos) < R_C for e in all_entities])
        
        # ================= 第一部分：构建节点特征 (带迷雾遮挡) =================
        for i, entity in enumerate(all_entities):
            type_id = 0.0 if isinstance(entity, Target) else (1.0 if isinstance(entity, Attacker) else 2.0)
            
            dist = np.linalg.norm(entity.state.p_pos - agent.state.p_pos)
            agent_idx = all_entities.index(agent)
            
            can_observe = False
            
            # 1. 目标基地和自己永远可见
            if entity == agent or isinstance(entity, Target):
                can_observe = True
            # 2. 当前视角是【防御者】
            elif isinstance(agent, Defender):
                if isinstance(entity, Defender):
                    if in_rc[i] or dist <= self.comm_radius_D:
                        can_observe = True
                elif isinstance(entity, Attacker):
                    if in_rc[i] or dist <= self.sensing_radius_D:
                        can_observe = True
            # 3. 当前视角是【攻击者】
            elif isinstance(agent, Attacker):
                if isinstance(entity, Attacker):
                    if dist <= self.comm_radius_A:
                        can_observe = True
                elif isinstance(entity, Defender):
                    if dist <= self.sensing_radius_A:
                        can_observe = True

            # 【执行迷雾遮挡】
            if can_observe:
                rel_pos_norm = (entity.state.p_pos - agent.state.p_pos) / (R_B * 2.0)
                rel_vel_norm = (entity.state.p_vel - agent.state.p_vel) / 12.0
            else:
                rel_pos_norm = np.zeros(2)
                rel_vel_norm = np.zeros(2)
            
            feat = np.concatenate([rel_pos_norm, rel_vel_norm, [type_id]])
            node_obs.append(feat)

        # ================= 第二部分：构建邻接矩阵 adj (GNN 的通信通道) =================
        # 给对角线填 1 (Self-loops)。GNN 必须有自环，否则在聚合信息时会遗忘自己的特征！
        np.fill_diagonal(adj, 1.0)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue
                
                e_i = all_entities[i]
                e_j = all_entities[j]
                dist_ij = np.linalg.norm(e_i.state.p_pos - e_j.state.p_pos)
                
                # 连边规则必须与论文中真实的物理通信/感知规则完全一致
                if isinstance(e_i, Target) or isinstance(e_j, Target):
                    pass  # 【修复1】：绝对不能设为 1.0！切断基地在 GNN 中的信息泄露通道！
                elif isinstance(e_i, Defender):
                    if isinstance(e_j, Defender):
                        # 【核心修复】：将错误的 and 改为 in_rc[j] 
                        if in_rc[j] or dist_ij <= self.comm_radius_D:
                            adj[i, j] = 1.0
                    elif isinstance(e_j, Attacker):
                        # 【核心修复】：只要攻击者在圈内或距离近，邻接矩阵就必须连通！
                        if in_rc[j] or dist_ij <= self.sensing_radius_D:
                            adj[i, j] = 1.0
                elif isinstance(e_i, Attacker):
                    if isinstance(e_j, Attacker):
                        if dist_ij <= self.comm_radius_A:
                            adj[i, j] = 1.0
                    elif isinstance(e_j, Defender):
                        if dist_ij <= self.sensing_radius_A:
                            adj[i, j] = 1.0

        # 返回时必须将 node_obs 转换为 float32 格式的 numpy 数组，解决 AttributeError 报错
        return np.array(node_obs, dtype=np.float32), adj

    def update_graph(self, world):
        if getattr(world, 'cached_dist_mag', None) is None:
            world.calculate_distances()
        dists = world.cached_dist_mag
        
        target = world.targets[0]
        all_entities = world.targets + world.attackers + world.defenders
        num_nodes = len(all_entities)
        
        in_rc = np.array([np.linalg.norm(e.state.p_pos - target.state.p_pos) < R_C for e in all_entities])
        
        row, col = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j: continue
                
                agent_i = all_entities[i]
                agent_j = all_entities[j]
                dist_ij = dists[i, j]
                edge_exists = False
                
                if isinstance(agent_j, Target) or isinstance(agent_i, Target):
                    pass  # 【修复2】：不要让 edge_exists 变成 True，切断边！
                elif isinstance(agent_i, Defender):
                    if isinstance(agent_j, Defender):
                        
                        if in_rc[j] or dist_ij <= self.comm_radius_D:
                            edge_exists = True
                    elif isinstance(agent_j, Attacker):
                        
                        if in_rc[j] or dist_ij <= self.sensing_radius_D:
                            edge_exists = True
                elif isinstance(agent_i, Attacker):
                    if isinstance(agent_j, Attacker):
                        if dist_ij <= self.comm_radius_A:
                            edge_exists = True
                    elif isinstance(agent_j, Defender):
                        if dist_ij <= self.sensing_radius_A:
                            edge_exists = True
                            
                if edge_exists:
                    row.append(i)
                    col.append(j)
                    
        world.edge_list = np.stack([row, col]) if len(row) > 0 else np.empty((2, 0), dtype=int)
        
        if hasattr(world, 'graph_feat_type') and world.edge_list.shape[1] > 0:
            world.edge_weight = dists[world.edge_list[0], world.edge_list[1]]
        

    def get_id(self, agent, world):
        all_entities = world.targets + world.attackers + world.defenders
        return np.array([all_entities.index(agent)], dtype=np.float32)
  
        
    def info_callback(self, agent, world):
        return {}