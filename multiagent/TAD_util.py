import numpy as np
from scipy.optimize import linprog
import copy
import warnings
  
# # other util functions
def Get_antiClockAngle(v1, v2):  # 向量v1逆时针转到v2所需角度。范围：0-2pi
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    
    # ================= 核心修复：防止 0 向量崩溃 ================= #
    if TheNorm < 1e-6:
        return 0.0
    # ============================================================== #
    
    # 叉乘 (修复浮点数越界导致 NaN 的隐患)
    cross_val = np.clip(np.cross(v1, v2) / TheNorm, -1.0, 1.0)
    rho = np.arcsin(cross_val)
    # 点乘
    cos_ = np.clip(np.dot(v1, v2) / TheNorm, -1.0, 1.0)
    theta = np.arccos(cos_)
    if rho < 0:
        return np.pi*2 - theta
    else:
        return theta

def Get_Beta(v1, v2):  
    # 规定逆时针旋转为正方向，计算v1转到v2夹角, -pi~pi
    # v2可能为0向量
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-4 or norm2 < 1e-4:
        # print('0 in denominator ')
        cos_ = 1  # 初始化速度为0，会出现分母为零
        return np.arccos(cos_)  # 0°
    
    cos_ = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    
    # 防止浮点数误差导致 cross>1.0 发生 arcsin 的 NaN
    cross_val = np.clip(np.cross(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    rho = np.arcsin(cross_val)
    
    theta = np.arccos(cos_)  # 输出的弧度范围[0,pi]
    if rho < 0:
        return -theta
    else:
        return theta

# ======== 保持原有的 APF/Deception 等工具函数不变 ======== #
def get_dist_cost(attacker, defender, target):
    '''
    cost based on distance and theta
    '''
    dist_coeff = 0.01 # tunable
    LOS_coeff = 0.5
    x_da = attacker.state.p_pos - defender.state.p_pos
    x_dt = target.state.p_pos - defender.state.p_pos
    dist_da = np.linalg.norm(x_da)
    dist_dt = np.linalg.norm(x_dt)

    e_da = x_da / dist_da
    e_dt = x_dt / dist_dt

    cos_theta = np.clip(np.dot(e_da, e_dt), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    if theta < np.pi/2:
        return dist_coeff * dist_da + LOS_coeff * theta
    else:
        return dist_coeff * dist_da + 5

def APF_defender(agent, attacker_):
    if attacker_.done:
        return np.array([0., 0.])
    
    # 简单的追逐逻辑，作为默认 APF 备份
    x_da = attacker_.state.p_pos - agent.state.p_pos
    dist_da = np.linalg.norm(x_da)
    if dist_da > 0:
        return (x_da / dist_da) * agent.max_accel if hasattr(agent, 'max_accel') else (x_da / dist_da) * agent.max_speed
    return np.array([0., 0.])

def APF_target(agent, attacker_, defender_):
    return np.array([0., 0.])

def attacker_policy(target, attacker):
    return np.array([0., 0.])

def update_fake_target(agent, world):
    pass

def GetAcuteAngle(v1, v2):
    # 计算两向量的锐角夹角 [0, pi/2]
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 < 1e-4 or norm2 < 1e-4:
        return 0.0
    cos_ = np.abs(np.dot(v1, v2)) / (norm1 * norm2)
    if cos_ > 1.0:
        cos_ = 1.0
    return np.arccos(cos_)

def calc_cost(attacker, defender, target):
    return 0.0

# TAD_util.py 底部添加
def map_defender_action(agent, raw_action):
    clipped_action = np.clip(raw_action[:2], -1.0, 1.0)
    return clipped_action * agent.max_accel

def map_attacker_action(agent, world, raw_action):
    w_v = np.clip(raw_action[0], -1.0, 1.0)
    xi_D = 0.5
    w_d = np.clip(raw_action[1], 0.0, 1.0) + xi_D
    w_t = np.clip(raw_action[2], -0.15, 1.35)
    
    target = world.targets[0]
    vec_to_target = target.state.p_pos - agent.state.p_pos
    dist_to_target = np.linalg.norm(vec_to_target)
    e_T = (vec_to_target / dist_to_target) if dist_to_target > 0 else np.zeros(2)
    
    eta_T = 1.0
    F_T = eta_T * agent.max_speed * e_T
    
    eta_D = 50000.0
    F_D = np.zeros(2)
    for defender in world.defenders:
        vec_from_def = agent.state.p_pos - defender.state.p_pos
        dist_from_def = np.linalg.norm(vec_from_def)
        
        # 修改 TAD_util.py 中的 F_D 计算
        if dist_from_def <= 25.0: 
            edge_dist = dist_from_def - agent.size - defender.size
            # 【直接删掉 if edge_dist > 0:】
            edge_dist_safe = max(edge_dist, 0.1) # 只要重叠，强行按照 0.1m 的极限距离计算
            rho_D_real = 25.0 - agent.size - defender.size 
            mag = eta_D * (1.0 / edge_dist_safe - 1.0 / rho_D_real) * (1.0 / edge_dist_safe**2)
            F_D += (vec_from_def / dist_from_def) * mag
                
    eta_A = 1000.0
    F_A = np.zeros(2)
    for other_attacker in world.attackers:
        if other_attacker is agent: continue
        vec_from_other = agent.state.p_pos - other_attacker.state.p_pos
        dist_other = np.linalg.norm(vec_from_other)
        
        # 【修复2】：修复你上一轮忘了改的攻击者撞车 Bug，同样受限于雷达并使用边缘距离
        if dist_other <= 25.0:
            edge_dist_A = dist_other - agent.size - other_attacker.size
            if edge_dist_A > 0:
                edge_dist_safe_A = max(edge_dist_A, 0.1)
                rho_A_real = 25.0 - agent.size - other_attacker.size
                mag = eta_A * (1.0 / edge_dist_safe_A - 1.0 / rho_A_real) * (1.0 / edge_dist_safe_A**2)
                F_A += (vec_from_other / dist_other) * mag

    R_A_sen = 25.0
    num_rays_half = 15 
    theta_target = np.arctan2(vec_to_target[1], vec_to_target[0])
    
    angles_l = np.linspace(theta_target, theta_target + np.pi/2, num_rays_half)
    angles_r = np.linspace(theta_target - np.pi/2, theta_target, num_rays_half)
    
    # ================= 动态计算阿波罗尼斯比例 =================
    actual_v_d = world.defenders[0].max_speed if len(world.defenders) > 0 else 3.0
    actual_v_a = agent.max_speed
    
    lam = actual_v_d / actual_v_a
    lam_sq = lam ** 2

    def get_free_rate(angles):
        N_hit = 0
        for angle in angles:
            u_vec = np.array([np.cos(angle), np.sin(angle)])
            hit = False
            for def_agent in world.defenders:
                vec_ad = def_agent.state.p_pos - agent.state.p_pos
                dist_ad = np.linalg.norm(vec_ad)
                if dist_ad > R_A_sen or dist_ad < 1e-4: continue
                
                center_rel = vec_ad / (1.0 - lam_sq) 
                R_O = (np.sqrt(lam_sq) / (1.0 - lam_sq)) * dist_ad
                # 【修复3】：物理装甲厚度保底！防穿模！
                R_O_effective = max(R_O, agent.size + def_agent.size)
                
                proj_len = np.dot(center_rel, u_vec)
                if proj_len > 0:
                    dist_line_sq = np.dot(center_rel, center_rel) - proj_len**2
                    if dist_line_sq <= R_O_effective**2:
                        hit = True
                        break
            if hit: N_hit += 1
        return 1.0 - (N_hit / num_rays_half)
    
    r_A_l = get_free_rate(angles_l)
    r_A_r = get_free_rate(angles_r)
    
    eta_V = 0.75
    e_T_perp_L = np.array([-e_T[1], e_T[0]]) 
    e_T_perp_R = np.array([e_T[1], -e_T[0]]) 
    
    # ================= 破局修复：空旷地带强制取消涡旋力 =================
    # 如果视野内根本没有防御者阻挡（100% 空旷），彻底关闭涡旋力
    if r_A_l == 1.0 and r_A_r == 1.0:
        F_V = np.array([0.0, 0.0])
    else:
        F_V = eta_V * agent.max_speed * ((r_A_l ** 2) * e_T_perp_L + (r_A_r ** 2) * e_T_perp_R)
    # ====================================================================
        
    combined_force = w_v * F_V + w_d * F_D + w_t * F_T + F_A
    
    norm_force = np.linalg.norm(combined_force)
    if norm_force > 0:
        return (combined_force / norm_force) * agent.max_accel
    return np.zeros(2)