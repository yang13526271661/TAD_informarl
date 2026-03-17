import numpy as np
import seaborn as sns
from .TAD_util import *

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # physical angle
        self.phi = 0  # 0-2pi
        # physical angular velocity
        self.p_omg = 0
        self.last_a = np.array([0, 0])
        # norm of physical velocity
        self.V = 0
        # 控制量（非加速度）：只需记录target，以便求attacker的policy_u
        self.controller = 0

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # index among all entities (important to set for distance caching)
        self.i = 0
        # name
        self.name = ''
        # properties:
        self.size = 1.0
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # entity can pass through non-hard walls
        self.ghost = False
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.done = False


class Target(Agent):
    def __init__(self):
        super(Target, self).__init__()


class Attacker(Agent):
    def __init__(self):
        super(Attacker, self).__init__()
        # self.assign_target = -1


class Defender(Agent):
    def __init__(self):
        super(Defender, self).__init__()


# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.targets = []
        self.attackers = []
        self.defenders = []

        self.walls = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # cache distances between all agents (not calculated by default)
        self.cache_dists = False
        self.cached_dist_vect = None
        self.cached_dist_mag = None

        self.world_length = 200

        self.attacker_belief = []
        self.cnt_dead = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def calculate_distances(self):
        if self.cached_dist_vect is None:
            # initialize distance data structure
            self.cached_dist_vect = np.zeros((len(self.entities),
                                              len(self.entities),
                                              self.dim_p))
            # calculate minimum distance for a collision between all entities
            self.min_dists = np.zeros((len(self.entities), len(self.entities)))
            for ia, entity_a in enumerate(self.entities):
                for ib in range(ia + 1, len(self.entities)):
                    entity_b = self.entities[ib]
                    min_dist = entity_a.size + entity_b.size
                    self.min_dists[ia, ib] = min_dist
                    self.min_dists[ib, ia] = min_dist

        for ia, entity_a in enumerate(self.entities):
            for ib in range(ia + 1, len(self.entities)):
                entity_b = self.entities[ib]
                delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
                self.cached_dist_vect[ia, ib, :] = delta_pos
                self.cached_dist_vect[ib, ia, :] = -delta_pos

        self.cached_dist_mag = np.linalg.norm(self.cached_dist_vect, axis=2)

        self.cached_collisions = (self.cached_dist_mag <= self.min_dists)

    def assign_agent_colors(self):
        n_dummies = 0
        if hasattr(self.agents[0], 'dummy'):
            n_dummies = len([a for a in self.agents if a.dummy])
        n_adversaries = 0
        if hasattr(self.agents[0], 'adversary'):
            n_adversaries = len([a for a in self.agents if a.adversary])
        n_good_agents = len(self.agents) - n_adversaries - n_dummies
        # sns.color_palette("OrRd", 10)
        dummy_colors = [(0.25, 0.25, 0.25)] * n_dummies
        adv_colors = sns.color_palette("OrRd", n_adversaries)
        good_colors = sns.color_palette("GnBu", n_good_agents)
        colors = dummy_colors + adv_colors + good_colors
        for color, agent in zip(colors, self.agents):
            agent.color = color

    def assign_landmark_colors(self):
        for landmark in self.landmarks:
            landmark.color = np.array([0.25, 0.25, 0.25])

    def assign_target_colors(self):
        for target in self.targets:
            target.color = np.array([0., 1., 0.])

    def assign_attacker_colors(self):
        for attacker in self.attackers:
            attacker.color = np.array([1., 0., 0.])

    def assign_defender_colors(self):
        for defender in self.defenders:
            defender.color = np.array([0., 0., 1.])

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action.u = agent.action_callback(agent, self)
            agent.action.c = np.zeros(self.dim_c)

        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)

        # calculate and store distances between all entities
        if self.cache_dists:
            self.calculate_distances()

    # gather agent action forces
    def apply_action_force(self, u):
        # set applied forces
        for i, agent in enumerate(self.agents):
            # ================= 核心修复：防止动作未分配 ================= #
            if agent.action.u is None:
                u[i] = np.zeros(self.dim_p)
            else:
                u[i] = agent.action.u
                if hasattr(u[i], "shape") and u[i].shape != (2,):
                    print(f"!!! agent {agent.name} has u of shape {u[i].shape} !!!")
            # ========================================================== #
        return u

    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a):
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None):
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None):
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        # wall collisions
        for a, entity_a in enumerate(self.agents):
            for wall in self.walls:
                f_a = self.get_wall_collision_force(entity_a, wall)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
        return p_force

    def integrate_state(self, u):  
        for i, agent in enumerate(self.agents):
            # [新增防线]：如果这个智能体是不可移动的（比如 Target），直接跳过物理计算
            if not agent.movable:
                continue

            if u[i] is None:
                u[i] = np.zeros(self.dim_p)

            v = agent.state.p_vel + u[i] * self.dt
            
            # [安全修复]：加上 agent.max_speed is not None 的判断
            if agent.max_speed is not None and np.linalg.norm(v) > agent.max_speed:
                v = v / np.linalg.norm(v) * agent.max_speed
            
            # [极端安全] 绝对兜底：禁止直接炸到无穷大
            v = np.nan_to_num(v, nan=0.0, posinf=agent.max_speed if agent.max_speed else 10.0, neginf=-(agent.max_speed if agent.max_speed else 10.0))

            if agent.done:
                agent.state.p_vel = np.array([0, 0]) # 速度清零
            else:
                v_x, v_y = v[0], v[1]
                theta = np.arctan2(v_y, v_x)
                if theta < 0:
                    theta += np.pi * 2
                
                delta_theta = Get_antiClockAngle(v, agent.state.p_vel)
                if delta_theta > np.pi:
                    delta_theta = delta_theta - np.pi * 2
                agent.state.p_omg = delta_theta / self.dt

                agent.state.phi = theta
                agent.state.last_a = u[i]
                agent.state.V = np.linalg.norm(v)
                agent.state.p_vel = v
                agent.state.p_pos += agent.state.p_vel * self.dt

    def update_agent_state(self, agent):
        pass

    # 修改 TAD_core.py 中的 get_collision_force
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  
        if (entity_a is entity_b):
            return [None, None]  
            
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.linalg.norm(delta_pos)
        
        if dist == 0.0:
            dist = 1e-5
            delta_pos = np.random.uniform(-1e-5, 1e-5, self.dim_p)
            
        # ================= 核心修复：分离方向与距离 =================
        # 1. 严格提取【单位方向向量】(长度永远为 1.0)
        unit_dir = delta_pos / dist  
        
        # 2. 计算穿透深度 (这里才使用 dist_safe 来防止对数爆炸)
        dist_safe = max(dist, 0.1) 
        dist_min = entity_a.size + entity_b.size
        k = self.contact_margin
        
        penetration = np.logaddexp(0, -(dist_safe - dist_min)/k)*k
        
        # 3. 最终受力 = 刚度 * 单位方向 * 穿透深度
        force = self.contact_force * unit_dir * penetration
        # ==============================================================
        
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # get collision forces for contact between an entity and a wall
    def get_wall_collision_force(self, entity, wall):
        if entity.ghost and not wall.hard:
            return None
        if wall.orient == 'H':
            prll_dim = 0
            perp_dim = 1
        else:
            prll_dim = 1
            perp_dim = 0
        ent_pos = entity.state.p_pos
        if (ent_pos[prll_dim] < wall.endpoints[0] - entity.size or
                ent_pos[prll_dim] > wall.endpoints[1] + entity.size):
            return None
        dist = ent_pos[perp_dim] - wall.axis_pos
        ent_min_dist = wall.width + entity.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(abs(dist) - ent_min_dist)/k)*k
        force = self.contact_force * penetration * np.sign(dist)
        if prll_dim == 0:
            force_vector = np.array([0, force])
        else:
            force_vector = np.array([force, 0])
        return force_vector