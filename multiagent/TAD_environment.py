import csv
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from multiagent.TAD_core import Agent
from onpolicy import global_var as glv
from multiagent.TAD_util import GetAcuteAngle
from multiagent.TAD_util import map_attacker_action, map_defender_action

# update bounds to center around agent
cam_range = 150
INFO = []  # render时可视化数据用

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, update_belief=None, 
                 post_step_callback=None, shared_viewer=True, 
                 discrete_action=False):
        
        # set CL
        self.use_policy = 1
        self.use_CL = 0
        self.CL_ratio = 0
        self.Cp = 0.6  
        self.JS_thre = 0

        # terminate
        self.is_ternimate = False

        self.world = world
        self.world_length = self.world.world_length
        self.current_step = 0
        
        # 自动识别强化学习策略智能体：支持在 attacker 和 defender 之间切换
        self.train_mode = getattr(self.world, 'train_mode', 'attacker')
        if self.train_mode == 'attacker':
            self.agents = self.world.attackers
        else:
            self.agents = self.world.defenders
        self.n = len(self.agents)
        
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback  
        self.post_step_callback = post_step_callback
        self.update_belief = update_belief

        self.discrete_action_space = discrete_action
        self.discrete_action_input = False
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        
        for agent in self.agents:
            # 直接使用 Box 连续空间控制物理运动
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-1.0, high=+1.0, shape=(self.world.dim_p,), dtype=np.float32)
            self.action_space.append(u_action_space)
            
            # observation space
            obs_dim = len(observation_callback(agent, self.world)) 
            share_obs_dim += obs_dim  
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        for i, agent in enumerate(self.agents):
            if 'attacker' in agent.name and getattr(self.world, 'train_mode', 'attacker') == 'attacker':
                # 【核心修复】：赋予攻击者左躲右闪的完整动力学边界
                # 严格对齐论文: w_v[-1, 1], w_d[0, 1], w_t[-0.15, 1.35]
                low_bound = np.array([-1.0, 0.0, -0.15], dtype=np.float32)
                high_bound = np.array([1.0, 1.0, 1.35], dtype=np.float32)
                self.action_space[i] = spaces.Box(low=low_bound, high=high_bound, shape=(3,), dtype=np.float32)

        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]
        
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n):
        self.current_step += 1
        obs_n = []
        reward_n = []
        done_n = []
        info_n = []
        start_ratio = 0.80
        self.JS_thre = int(self.world_length*start_ratio*set_JS_curriculum(self.CL_ratio/self.Cp))

        # set action for poliy agents
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        
        # advance world state
        self.world.step()

        # record observation for each agent
        for i, agent in enumerate(self.agents):
            if agent.done:
                # 死亡后强制零动作，避免残余隐状态干扰物理
                agent.action.u = np.zeros(self.world.dim_p)
                
            obs_n.append(self._get_obs(agent))
            reward_n.append([self._get_reward(agent)])
            done_n.append(self._get_done(agent))
            info = {'individual_reward': self._get_reward(agent)}
            env_info = self._get_info(agent)
            if 'fail' in env_info.keys():
                info['fail'] = env_info['fail']
            info_n.append(info)

        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [[reward]] * self.n

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        terminate = []
        current_dead = 0
        attacker_belief = []
        
        terminate = []
        for agent in self.world.agents:
            # 修复 1：你的 target 名字叫 'target 0'，所以必须用 'in' 来匹配
            if 'target' in agent.name:
                terminate.append(agent.done)
            if agent.name=='attacker':
                # 兼容旧代码，防止报错
                belief = getattr(agent, 'fake_target', -1)
                attacker_belief.append(belief)
                agent.last_belief = belief
                agent.last_lock = getattr(agent, 'is_locked', False)
                
            if agent.done:
                current_dead += 1

        # 修复 2：只有在真正找到 target 的情况下，才去判断是否全灭
        if len(terminate) > 0:
            self.is_ternimate = True if all(terminate) else False
        else:
            self.is_ternimate = False
            
        if self.is_ternimate:
            done_n = [True] * self.n
            
        if self.update_belief is not None and not all(done_n):
            if (self.current_step%15==0 and self.current_step < 180) or current_dead > self.world.cnt_dead:
                self.update_belief(self.world)

        self.world.cnt_dead = current_dead
        self.world.attacker_belief = attacker_belief

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.current_step = 0
        self.reset_callback(self.world)
        self._reset_render()
        
        obs_n = []
        # 同步更新 agents
        if getattr(self.world, 'train_mode', 'attacker') == 'attacker':
            self.agents = self.world.attackers
        else:
            self.agents = self.world.defenders

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

        return obs_n

    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    def _get_done(self, agent):
        if self.done_callback is None:
            # 没有外部 done 回调时，同时考虑物理标记的 agent.done，避免终止状态继续积累梯度与隐藏态
            return agent.done or (self.current_step >= self.world_length)
        return self.done_callback(agent, self.world)

    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)


    def _set_action(self, action, agent, action_space):
        # 1. 基础清零
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        # 2. 【核心修复】：防止“诈尸”！如果智能体已阵亡或出界，直接返回，忽略神经网络的输出
        if getattr(agent, 'done', False):
            return 

        # 3. 解析网络动作
        action_args = action[0] if isinstance(action, list) else action

        # 4. 【更清晰的写法】：直接根据 agent 的身份调用统一映射器
        if 'attacker' in agent.name:
            # 无论是主角在训练，还是未来扩展，只要它是攻击者，就用这套物理法则
            agent.action.u = map_attacker_action(agent, self.world, action_args)
            
        elif 'defender' in agent.name:
            # 只要它是防守者，就用这套物理法则
            agent.action.u = map_defender_action(agent, action_args)


    def _set_CL(self, CL_ratio):
        glv.set_value('CL_ratio', CL_ratio)
        self.CL_ratio = glv.get_value('CL_ratio')
        # 同步写回 world，确保 reset_world 能读取到最新的课程比率
        if hasattr(self, 'world'):
            self.world.CL_ratio = self.CL_ratio

    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    def render(self, mode='human', close=False):
        if close:
            for i, viewer in enumerate(self.viewers):
                if viewer is not None:
                    viewer.close()
                self.viewers[i] = None
            return []
        if mode == 'human':
            pass
            
        for i in range(len(self.viewers)):
            if self.viewers[i] is None:
                from . import rendering
                self.viewers[i] = rendering.Viewer(700, 700)
                
        if self.render_geoms is None:
            from . import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            self.line = {}
            self.comm_geoms = []
            for entity in self.world.entities:
                # ================= 核心修复 1：使用真实的 entity.size ================= #
                geom = rendering.make_circle(entity.size)
                # ==================================================================== #
                xform = rendering.Transform()
                entity_comm_geoms = []
                
                # 区分背景区域（半透明）和智能体实体（不透明）
                if 'target' in entity.name or 'attacker' in entity.name or 'defender' in entity.name:
                    geom.set_color(*entity.color, alpha=1.0)
                else:
                    geom.set_color(*entity.color, alpha=0.35) # 背景光环高透明度
                    
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
                self.comm_geoms.append(entity_comm_geoms)
            
            for wall in getattr(self.world, 'walls', []):
                pass 
                
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)
                    
        results = []
        for i in range(len(self.viewers)):
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(-120, 120, -120, 120)
            
            # csv save logging
            data_ = ()
            for j, attacker in enumerate(self.world.attackers):
                # 增加 getattr 兼容，防止没有分层欺骗变量时报错
                data_ = data_ + (j, attacker.state.p_pos[0], attacker.state.p_pos[1],
                                 attacker.state.p_vel[0], attacker.state.p_vel[1],
                                 getattr(attacker, 'true_target', -1), getattr(attacker, 'fake_target', -1), getattr(attacker, 'is_locked', False), attacker.done)
            for j, target in enumerate(self.world.targets):
                data_ = data_ + (j, target.state.p_pos[0], target.state.p_pos[1],
                                 target.state.p_vel[0], target.state.p_vel[1], target.done)
            for j, defender in enumerate(self.world.defenders):
                data_ = data_ + (j, defender.state.p_pos[0], defender.state.p_pos[1],
                                 defender.state.p_vel[0], defender.state.p_vel[1], getattr(defender, 'attacker', -1), defender.done)
            INFO.append(data_)

            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                self.line[e] = self.viewers[i].draw_line(entity.state.p_pos, entity.state.p_pos+entity.state.p_vel*1.0)
                
                # ================= 核心修复 2：逐帧维持正确的透明度 ================= #
                if 'target' in entity.name or 'attacker' in entity.name or 'defender' in entity.name:
                    self.render_geoms[e].set_color(*entity.color, alpha=1.0)
                    self.line[e].set_color(*entity.color, alpha=1.0)
                else:
                    self.render_geoms[e].set_color(*entity.color, alpha=0.35)
                    self.line[e].set_color(*entity.color, alpha=0.0) # 背景圈不需要速度线
                # ================================================================== #

            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

def set_JS_curriculum(CL_ratio):
    k = 2.0
    delta = 1-(np.exp(-k*(-1))-np.exp(k*(-1)))/(np.exp(-k*(-1))+np.exp(k*(-1)))
    x = 2*CL_ratio-1
    y_mid = (np.exp(-k*x)-np.exp(k*x))/(np.exp(-k*x)+np.exp(k*x))-delta*x**3
    func_ = (y_mid+1)/2
    return func_

class MultiAgentGraphEnv(MultiAgentEnv):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        world,
        reset_callback=None,
        reward_callback=None,
        observation_callback=None,
        graph_observation_callback=None,
        id_callback=None,
        info_callback=None,
        done_callback=None,
        update_belief=None,
        post_step_callback=None,
        update_graph=None,
        shared_viewer=True,
        discrete_action=False,
    ) -> None:
        super(MultiAgentGraphEnv, self).__init__(
            world, reset_callback, reward_callback, observation_callback, info_callback,
            done_callback, update_belief, post_step_callback, shared_viewer, discrete_action,
        )
        self.update_graph = update_graph
        self.graph_observation_callback = graph_observation_callback
        self.id_callback = id_callback
        self.set_graph_obs_space()

    def set_graph_obs_space(self):
        self.node_observation_space = []
        self.adj_observation_space = []
        self.edge_observation_space = []
        self.agent_id_observation_space = []
        self.share_agent_id_observation_space = []
        num_agents = len(self.agents)
        for agent in self.agents:
            node_obs, adj = self.graph_observation_callback(agent, self.world)
            node_obs_dim = node_obs.shape  
            adj_dim = adj.shape  
            edge_dim = 1  
            agent_id_dim = 1  
            
            self.node_observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=node_obs_dim, dtype=np.float32))
            self.adj_observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=adj_dim, dtype=np.float32))
            self.edge_observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(edge_dim,), dtype=np.float32))
            self.agent_id_observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(agent_id_dim,), dtype=np.float32))
            self.share_agent_id_observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(num_agents * agent_id_dim,), dtype=np.float32))

    def step(self, action_n):
        if self.update_graph is not None:
            self.update_graph(self.world)
        self.current_step += 1
        # ================= 核心修复：将环境的时间步同步给物理世界 =================
        self.world.current_step = self.current_step

        obs_n, reward_n, done_n, info_n = [], [], [], []
        node_obs_n, adj_n, agent_id_n = [], [], []

        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])

        self.world.step()

        done_check = []
        for i, agent in enumerate(self.agents):
            if agent.done:
                agent.action.u = np.zeros(self.world.dim_p)
                
            obs_n.append(self._get_obs(agent))
            agent_id_n.append(self._get_id(agent))
            
            node_obs, adj = self._get_graph_obs(agent)
            node_obs_n.append(node_obs)
            adj_n.append(adj)
            
            reward = self._get_reward(agent)
            reward_n.append([reward])
            
            done_status = self._get_done(agent)
            done_n.append(done_status)
            done_check.append(agent.done)
            
            info = {'individual_reward': reward}
            info_n.append(info)

        if self.shared_reward:
            reward_sum = np.sum(reward_n)
            reward_n = [[reward_sum]] * self.n

        if self.post_step_callback is not None:
            self.post_step_callback(self.world)

        terminate_target = [a.done for a in self.world.targets]
        terminate_attacker = [a.done for a in self.world.attackers]
        
        # 只要目标被摧毁，或者所有攻击者都已阵亡，立即终止本局！
        if (len(terminate_target) > 0 and all(terminate_target)) or (len(terminate_attacker) > 0 and all(terminate_attacker)):
            self.is_ternimate = True
        else:
            self.is_ternimate = False
            
        if self.is_ternimate:
            done_n = [True] * self.n

        return obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n

    def reset(self):
        self.current_step = 0
        # ================= 核心修复：回合重置时，同步清零物理世界的时间步 =================
        self.world.current_step = self.current_step
        self.reset_callback(self.world)
        self._reset_render()
        
        obs_n, node_obs_n, adj_n, agent_id_n = [], [], [], []
        # 同步更新 GNN agents
        if getattr(self.world, 'train_mode', 'attacker') == 'attacker':
            self.agents = self.world.attackers
        else:
            self.agents = self.world.defenders

        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            agent_id_n.append(self._get_id(agent))
            node_obs, adj = self._get_graph_obs(agent)
            node_obs_n.append(node_obs)
            adj_n.append(adj)

        return obs_n, agent_id_n, node_obs_n, adj_n

    def _get_graph_obs(self, agent: Agent):
        if self.graph_observation_callback is None:
            return None, None
        return self.graph_observation_callback(agent, self.world)

    # ================= 修改的核心位置：向回调传递 self.world ================= #
    def _get_id(self, agent: Agent):
        if self.id_callback is None:
            return None
        return self.id_callback(agent, self.world) 
    # ========================================================================= #