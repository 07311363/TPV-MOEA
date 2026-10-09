"""
STT-DMOEA 响应策略模块
包含环境变化响应的核心函数
"""
import numpy as np
from gcn_sort.inference import rank_with_gcn


def boundary_check(x_var, low_bounds, upp_bounds):
    """
    边界检查函数
    如果变量超出边界，则将其重置到边界范围内的随机位置
    
    Args:
        x_var: 决策变量数组
        low_bounds: 下边界数组
        upp_bounds: 上边界数组
    
    Returns:
        x_var: 边界检查后的决策变量
    """
    x_var = np.array(x_var, dtype=np.float64)
    low_bounds = np.array(low_bounds, dtype=np.float64)
    upp_bounds = np.array(upp_bounds, dtype=np.float64)
    
    for i in range(len(x_var)):
        if x_var[i] < low_bounds[i]:
            x_var[i] = low_bounds[i] + np.random.random() * (upp_bounds[i] - low_bounds[i]) / 2.0
        if x_var[i] > upp_bounds[i]:
            x_var[i] = (low_bounds[i] + upp_bounds[i]) / 2.0 + np.random.random() * (upp_bounds[i] - low_bounds[i]) / 2.0
    
    return x_var

def _indiv_differentOJ(vec1, vec2):
    "欧氏距离计算"
    dist = 0.0
    for i in range(len(vec1)):
        diff = vec1[i] - vec2[i]
        dist += diff * diff
    return np.sqrt(dist)

def _find_region_for_point(f_point, regions, nobj):
    """
    在区域列表中为给定目标向量找到所属区域（第一个满足 A<=f<=B 的区域）
    """
    for idx, reg in enumerate(regions):
        A = reg["A"]
        B = reg["B"]
        if A.size == 0 or B.size == 0:
            continue
        if np.all((f_point >= A) & (f_point <= B)):
            return idx
    return None


import heapq
import numpy as np


import heapq
import numpy as np


def detect_shared_points(v_prime_obj, nd_obj, nobj):

    boundary_idx = []
    shared_idx = []

    if v_prime_obj.size == 0 or nd_obj.size == 0:
        return boundary_idx, shared_idx

    # =========================================================
    # 1. 计算每个点到 ND 的最小距离
    # =========================================================
    min_distances = []

    for i in range(v_prime_obj.shape[0]):

        min_dist = 1e100

        for j in range(nd_obj.shape[0]):
            dis = _indiv_differentOJ(v_prime_obj[i], nd_obj[j])
            if dis < min_dist:
                min_dist = dis

        min_distances.append((min_dist, i))

    # （降序处理）
    min_distances.sort(key=lambda x: x[0], reverse=True)

    # =========================================================
    # 2. 逐点处理
    # =========================================================
    for _, point_idx in min_distances:

        current_point = v_prime_obj[point_idx]

        # =====================================================
        # 3. 构建 ND 距离集合
        # =====================================================
        nd_heap = []

        for j in range(nd_obj.shape[0]):
            dis = _indiv_differentOJ(current_point, nd_obj[j])
            nd_heap.append((dis, j))

        nd_heap.sort(key=lambda x: x[0])

        # =====================================================
        # 4. 取最近 4 个 ND
        # =====================================================
        k = min(4, len(nd_heap))

        direction_vectors = []

        for t in range(k):
            _, nd_idx = nd_heap[t]

            vec = nd_obj[nd_idx] - current_point
            direction_vectors.append(vec)

        # =====================================================
        # 5. angle check
        # =====================================================
        is_crosspoint = False

        num_vectors = len(direction_vectors)

        for ii in range(num_vectors):
            for jj in range(ii + 1, num_vectors):

                vec1 = direction_vectors[ii]
                vec2 = direction_vectors[jj]

                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 1e-12 and norm2 > 1e-12:

                    cos_theta = dot_product / (norm1 * norm2)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)

                    angle = np.arccos(cos_theta) * 180.0 / np.pi

                    if angle > 90.0:
                        is_crosspoint = True
                        break

            if is_crosspoint:
                break

        if is_crosspoint:
            shared_idx.append(point_idx)

    # =========================================================
    # 6. extreme points
    # =========================================================
    for obj in range(nobj):

        extreme_idx = -1
        extreme_val = 1e100

        for i in range(v_prime_obj.shape[0]):

            val = v_prime_obj[i, obj]

            if val < extreme_val:
                extreme_val = val
                extreme_idx = i

        if extreme_idx not in shared_idx:
            shared_idx.append(extreme_idx)

        boundary_idx.append(extreme_idx)

    # =========================================================
    # 7. 去重 + 排序
    # =========================================================
    boundary_idx = sorted(list(set(boundary_idx)))
    shared_idx = sorted(list(set(shared_idx)))

    shared_idx = sorted(shared_idx, key=lambda x: v_prime_obj[x, 0])

    return boundary_idx, shared_idx


def build_regions(shared_idx_sorted, v_prime_obj, v_prime_dec, nd_obj, nobj, 
                  Dt_raw, prev_pop_dec, nvar, moead, gen,
                  low_bounds=None, upp_bounds=None,
                  ref_bind_mode='scaled_weight'):
    """
    构建区域并分配nd点

    规则：
    - 2目标：每个区域由相邻2个共享点定义
    - 3目标：每个区域由相邻3个共享点定义
    """
    regions = []
    assigned_nd = set()  # 防止重复分配

    if shared_idx_sorted.size == 0:
        return regions

    if nobj == 3:
        window_size = 3
    else:
        window_size = 2

    if shared_idx_sorted.size < window_size:
        return regions

    for rr in range(shared_idx_sorted.size - window_size + 1):
        idx_group = np.array(shared_idx_sorted[rr: rr + window_size], dtype=int)

        pts_obj = v_prime_obj[idx_group, :]
        A_bounds = np.min(pts_obj, axis=0)
        B_bounds = np.max(pts_obj, axis=0)
        origin = A_bounds
        vec = B_bounds - A_bounds
        nrm = np.linalg.norm(vec)
        unit = vec / nrm if nrm > 0 else vec

        try:
            if ref_bind_mode == 'scaled_weight':
                W_global = moead.weight_vector()
                objs_list = []
                if v_prime_obj is not None and v_prime_obj.size > 0:
                    objs_list.append(v_prime_obj)
                if nd_obj is not None and nd_obj.size > 0:
                    objs_list.append(nd_obj)
                if len(objs_list) > 0:
                    all_objs = np.vstack(objs_list)
                    gmin = np.min(all_objs, axis=0)
                    gmax = np.max(all_objs, axis=0)
                else:
                    gmin = A_bounds
                    gmax = B_bounds
                gspan = gmax - gmin
                scale_vec = B_bounds - A_bounds
                ratio = np.ones_like(scale_vec)
                mask = gspan > 1e-12
                ratio[mask] = scale_vec[mask] / gspan[mask]
                W_local = W_global * ratio
                sums = np.sum(W_local, axis=1, keepdims=True)
                sums[sums <= 1e-12] = 1.0
                W_local_l1 = W_local / sums
                norms = np.linalg.norm(W_local_l1, axis=1)
                valid = norms > 1e-12
                weight_dirs = (W_local_l1[valid] / norms[valid][:, None]) if np.any(valid) else np.empty((0, nobj))
            elif ref_bind_mode == 'no_weight_scale':
                W_global = moead.weight_vector()
                sums = np.sum(W_global, axis=1, keepdims=True)
                sums[sums <= 1e-12] = 1.0
                W_global_l1 = W_global / sums
                norms = np.linalg.norm(W_global_l1, axis=1)
                valid = norms > 1e-12
                weight_dirs = (W_global_l1[valid] / norms[valid][:, None]) if np.any(valid) else np.empty((0, nobj))
                ratio = np.ones_like(B_bounds - A_bounds)
            else:
                # no_weight_scale: 不做权重向量缩放/方向绑定
                weight_dirs = np.empty((0, nobj))
                ratio = np.ones_like(B_bounds - A_bounds)
        except Exception:
            weight_dirs = np.empty((0, nobj))
            ratio = np.ones_like(B_bounds - A_bounds)

        # nd成员：判定每个nd点是否所有维度都在[A_bounds, B_bounds]区间内
        nd_idx = []
        if nd_obj.size > 0:
            for nd_i in range(nd_obj.shape[0]):
                if nd_i in assigned_nd:
                    continue
                in_region = True
                for dim in range(nobj):
                    if not (A_bounds[dim] <= nd_obj[nd_i, dim] <= B_bounds[dim]):
                        in_region = False
                        break
                if in_region:
                    nd_idx.append(nd_i)
                    assigned_nd.add(nd_i)

        nd_idx = np.array(nd_idx, dtype=int)

        # 决策空间区域：由共享点组对应决策值共同决定
        if v_prime_dec.size > 0:
            pts_dec = v_prime_dec[idx_group, :]
            A_dec = np.min(pts_dec, axis=0)
            B_dec = np.max(pts_dec, axis=0)
        else:
            A_dec = None
            B_dec = None

        shifted_prev_pop_dec = None
        shifted_prev_pop_obj = None
        if A_dec is not None and B_dec is not None and prev_pop_dec is not None and prev_pop_dec.shape[0] > 0:
            shifted_pop_dec = prev_pop_dec + Dt_raw
            if low_bounds is not None and upp_bounds is not None:
                for idx in range(shifted_pop_dec.shape[0]):
                    shifted_pop_dec[idx] = boundary_check(shifted_pop_dec[idx], low_bounds, upp_bounds)
            else:
                shifted_pop_dec = np.clip(shifted_pop_dec, 0.0, 1.0)
            shifted_pop_obj = np.array([moead.problem(xd, gen) for xd in shifted_pop_dec])

            in_region_mask = np.all(
                (shifted_pop_obj >= A_bounds) & (shifted_pop_obj <= B_bounds),
                axis=1
            )
            shifted_prev_pop_dec = shifted_pop_dec[in_region_mask]
            shifted_prev_pop_obj = shifted_pop_obj[in_region_mask]

        regions.append({
            "pair_shared_idx": idx_group,
            "A": A_bounds,
            "B": B_bounds,
            "origin": origin,
            "unit_dir": unit,
            "members_vprime_idx": idx_group,
            "members_nd_idx": nd_idx,
            "A_dec": A_dec if A_dec is not None else np.array([]),
            "B_dec": B_dec if B_dec is not None else np.array([]),
            "shifted_prev_pop_dec": shifted_prev_pop_dec if shifted_prev_pop_dec is not None else np.empty((0, nvar)),
            "shifted_prev_pop_obj": shifted_prev_pop_obj if shifted_prev_pop_obj is not None else np.empty((0, nobj)),
            "weight_dirs": weight_dirs,
            "scale_ratio": ratio,
        })

    return regions


def find_reference_point(nd_obj_point, pool_obj, origin, mode='cosine'):
    """
    为给定的点在参考池中找到最匹配的参考点

    参数：
        nd_obj_point: 目标点的目标值
        pool_obj: 参考池的目标值矩阵
        origin: 区域原点
        mode: 绑定模式，'cosine'（方向相似）或 'euclidean'（纯欧氏最近）

    返回：
        ref_idx: 参考池中最匹配点的索引
    """
    if mode == 'euclidean':
        dists = np.linalg.norm(pool_obj - nd_obj_point, axis=1)
        return int(np.argmin(dists))

    # cosine模式：方向优先，异常时回退欧氏
    x_vec = nd_obj_point - origin
    x_norm = np.linalg.norm(x_vec)

    Y = pool_obj - origin
    y_norms = np.linalg.norm(Y, axis=1)

    cosines = np.full(Y.shape[0], -np.inf)
    for k in range(Y.shape[0]):
        if x_norm > 0 and y_norms[k] > 0:
            cosines[k] = np.dot(x_vec, Y[k]) / (x_norm * y_norms[k])

    if not np.isfinite(cosines).any():
        dists = np.linalg.norm(pool_obj - nd_obj_point, axis=1)
        ref_idx = int(np.argmin(dists))
    else:
        ref_idx = int(np.nanargmax(cosines))

    return ref_idx


def generate_local_random_nd(nd_dec, nd_obj, N, nvar, moead, gen, sigma=0.0, 
                               oldmin=None, oldmax=None, newmin=None, newmax=None,
                               low_bounds=None, upp_bounds=None):
    """
    生成"局部随机 + 当前ND"的个体
    1. 使用反射公式 rand2(2*newmin-oldmin, 2*newmax-oldmax) 生成局部随机个体
    2. 合并当前ND解
    3. 通过选择机制选择较优个体
    
    Args:
        nd_dec: 当前非支配解决策变量
        nd_obj: 当前非支配解目标值
        N: 目标种群大小
        nvar: 决策变量维度
        moead: MOEAD实例
        gen: 当前代数
        sigma: 扰动标准差
        oldmin, oldmax: 上一次环境的ND决策变量边界
        newmin, newmax: 当前环境的ND决策变量边界
        low_bounds: 全局决策空间下边界（可选）
        upp_bounds: 全局决策空间上边界（可选）
    
    Returns:
        local_random_decs: 生成的局部随机个体决策变量
        local_random_objs: 生成的局部随机个体目标值
    """
    local_random_decs = []
    local_random_objs = []
    
    # 策略1: 使用反射公式生成局部随机个体
    if newmin is not None and newmax is not None and oldmin is not None and oldmax is not None:
        for _ in range(N):
            new_dec = np.zeros(nvar)
            for j in range(nvar):
                low = 2.0 * newmin[j] - oldmin[j]
                high = 2.0 * newmax[j] - oldmax[j]
                new_dec[j] = np.random.uniform(low, high)
            # 边界检查
            if low_bounds is not None and upp_bounds is not None:
                new_dec = boundary_check(new_dec, low_bounds, upp_bounds)
            else:
                new_dec = np.clip(new_dec, 0.0, 1.0)
            new_obj = moead.problem(new_dec, gen)
            local_random_decs.append(new_dec)
            local_random_objs.append(new_obj)
    else:
        # 没有边界信息时，使用均匀随机生成
        for _ in range(N):
            new_dec = np.random.rand(nvar)
            # 边界检查
            if low_bounds is not None and upp_bounds is not None:
                new_dec = boundary_check(new_dec, low_bounds, upp_bounds)
            else:
                new_dec = np.clip(new_dec, 0.0, 1.0)
            new_obj = moead.problem(new_dec, gen)
            local_random_decs.append(new_dec)
            local_random_objs.append(new_obj)
    
    local_random_decs = np.array(local_random_decs)
    local_random_objs = np.array(local_random_objs)
    
    # 策略2: 添加当前ND解（仅合并决策变量）
    if nd_dec is not None and nd_dec.shape[0] > 0:
        local_random_decs = np.vstack([local_random_decs, nd_dec])

    # 按当前环境重新评估
    local_random_objs = np.array([moead.problem(xd, gen) for xd in local_random_decs]) 

    return local_random_decs, local_random_objs


def predict_population_with_regions(regions, sol, fitness, Dt, moead, gen, sigma=0.0,
                                   low_bounds=None, upp_bounds=None):
    """
    对整个种群进行绑定参考点 + 预测
    每个个体 sol[i] 都生成一个预测解 new_sol[i]，不再依赖 nd 数量

    Args:
        regions: 区域列表（由 build_regions 构建）
        sol: 当前种群决策变量 (N, nvar)
        fitness: 当前种群目标值 (N, nobj)
        Dt: 环境变化向量
        moead: MOEAD 实例
        gen: 当前代数
        sigma: 扰动标准差
        low_bounds: 全局决策空间下边界（可选）
        upp_bounds: 全局决策空间上边界（可选）

    Returns:
        new_sol: 预测后的种群决策变量 (N, nvar)
        new_fit: 预测后的种群目标值 (N, nobj)
    """
    N, nvar = sol.shape
    nobj = fitness.shape[1]
    new_sol = np.zeros_like(sol)
    new_fit = np.zeros_like(fitness)

    # 调试统计
    region_used_count = 0  # 成功使用区域预测的个体数
    fallback_count = 0     # 使用保底策略的个体数

    for i in range(N):
        x = sol[i, :].copy()
        f = fitness[i, :].copy()

        region_idx = _find_region_for_point(f, regions, nobj)
        used_region = False

        if region_idx is not None:
            reg = regions[region_idx]
            origin_reg = reg["origin"]
            pool_dec_shifted = reg["shifted_prev_pop_dec"]
            pool_obj_shifted = reg["shifted_prev_pop_obj"]
            weight_dirs = reg.get("weight_dirs", np.empty((0, nobj)))

            if pool_dec_shifted.shape[0] > 0:
                ref_idx = None
                if weight_dirs.shape[0] > 0:
                    # 以区域原点为锚，计算到每个权重方向直线的垂直距离，选择最近的方向
                    u = f - origin_reg
                    u2 = float(np.dot(u, u))
                    if u2 < 0:
                        u2 = 0.0
                    projs = weight_dirs.dot(u)  # (M,)
                    dists_sq = np.maximum(u2 - projs**2, 0.0)
                    w_best = int(np.argmin(dists_sq))
                    # 在参考池中，找对同一方向最近的点
                    V = pool_obj_shifted - origin_reg  # (P, m)
                    if V.shape[0] > 0:
                        v2 = np.einsum('ij,ij->i', V, V)
                        proj_pool = V.dot(weight_dirs[w_best])  # (P,)
                        d_pool_sq = np.maximum(v2 - proj_pool**2, 0.0)
                        ref_idx = int(np.argmin(d_pool_sq))

                # 兜底：若无可用方向或参考池为空，回退到余弦相似度/最近欧氏
                if ref_idx is None:
                    bind_mode = 'euclidean' if weight_dirs.shape[0] == 0 else 'cosine'
                    ref_idx = find_reference_point(f, pool_obj_shifted, origin_reg, mode=bind_mode)

                ref_dec_shifted = pool_dec_shifted[ref_idx, :]
                noise = np.random.normal(0, np.abs(sigma), nvar) if sigma is not None else 0.0
                # x + Dt + (x - ref) + noise
                H = x - ref_dec_shifted
                new_dec = x + Dt + H + noise
                region_used_count += 1  #
                # 边界检查：预测后的决策变量需要检查是否超出全局决策空间
                if low_bounds is not None and upp_bounds is not None:
                    new_dec = boundary_check(new_dec, low_bounds, upp_bounds)
                else:
                    new_dec = np.clip(new_dec, 0.0, 1.0)
                new_obj = moead.problem(new_dec, gen)

                new_sol[i, :] = new_dec
                new_fit[i, :] = new_obj
                used_region = True

        if not used_region:
            # 保底策略：x + Dt + 噪声
            noise = np.random.normal(0, np.abs(sigma), nvar) if sigma is not None else 0.0
            new_dec = x + Dt + noise
            # 边界检查：预测后的决策变量需要检查是否超出全局决策空间
            if low_bounds is not None and upp_bounds is not None:
                new_dec = boundary_check(new_dec, low_bounds, upp_bounds)
            else:
                new_dec = np.clip(new_dec, 0.0, 1.0)
            new_obj = moead.problem(new_dec, gen)
            new_sol[i, :] = new_dec
            new_fit[i, :] = new_obj
            fallback_count += 1  # 使用保底策略

    return new_sol, new_fit

def predict_population_dual_strategy(regions, sol, fitness, Dt, moead, gen, 
                                       nd_dec, nd_obj, sigma=0.0,
                                       oldmin=None, oldmax=None, newmin=None, newmax=None,
                                       low_bounds=None, upp_bounds=None):
    """
    双策略组合预测：一半来自"区域预测+Dt+H+高斯"，一半来自"局部随机+当前ND"
    1. 第一半：区域预测 + Dt + H + 高斯 选择N/2个
    2. 第二半：局部随机(反射公式) + 当前ND 选择N/2个
    
    Args:
        regions: 区域列表
        sol: 当前种群决策变量 (N, nvar)
        fitness: 当前种群目标值 (N, nobj)
        Dt: 环境变化向量
        moead: MOEAD实例
        gen: 当前代数
        nd_dec: 当前非支配解决策变量
        nd_obj: 当前非支配解目标值
        sigma: 扰动标准差
        oldmin, oldmax: 上一次环境的ND决策变量边界
        newmin, newmax: 当前环境的ND决策变量边界
        low_bounds: 全局决策空间下边界（可选）
        upp_bounds: 全局决策空间上边界（可选）
    
    Returns:
        new_sol: 预测后的种群决策变量 (N, nvar)
        new_fit: 预测后的种群目标值 (N, nobj)
    """
    N, nvar = sol.shape
    nobj = fitness.shape[1]
    half_n = N // 2
    
    # 策略1: 使用区域预测生成个体 (region prediction + Dt + H + Gaussian)
    region_sol, region_fit = predict_population_with_regions(
        regions, sol, fitness, Dt, moead, gen, sigma=sigma,
        low_bounds=low_bounds, upp_bounds=upp_bounds
    )
    
    # 策略2: 使用局部随机+当前ND生成个体 
    local_random_decs, local_random_objs = generate_local_random_nd(
        nd_dec, nd_obj, N, nvar, moead, gen, sigma=sigma,
        oldmin=oldmin, oldmax=oldmax, newmin=newmin, newmax=newmax,
        low_bounds=low_bounds, upp_bounds=upp_bounds
    )
    
    # 分别从两种策略生成的个体中各选择N/2个较优个体
    
    # 从区域预测策略中选择前N/2个个体
    if region_sol.shape[0] >= half_n:
        region_selected_decs, region_selected_objs = simple_nsga2_select(region_sol, region_fit, half_n)
    else:
        region_selected_decs, region_selected_objs = region_sol, region_fit
    
    # 从局部随机策略中选择前N/2个个体
    if local_random_decs.shape[0] >= half_n:
        local_selected_decs, local_selected_objs = simple_nsga2_select(local_random_decs, local_random_objs, half_n)
    else:
        local_selected_decs, local_selected_objs = local_random_decs, local_random_objs
    
    # 合并两个策略选择的个体
    selected_decs = np.vstack([region_selected_decs, local_selected_decs])
    selected_objs = np.vstack([region_selected_objs, local_selected_objs])

    # 如果合并后不足N个，补充随机个体
    if selected_decs.shape[0] < N:
        need = N - selected_decs.shape[0]
        random_decs = np.random.rand(need, nvar)
        random_decs = np.clip(random_decs, 0.0, 1.0)
        random_objs = np.array([moead.problem(xd, gen) for xd in random_decs])
        selected_decs = np.vstack([selected_decs, random_decs])
        selected_objs = np.vstack([selected_objs, random_objs])
    
    # 如果合并后超过N个，使用NSGA-II选择到N个
    elif selected_decs.shape[0] > N:
        selected_decs, selected_objs = simple_nsga2_select(selected_decs, selected_objs, N)
    
    return selected_decs, selected_objs


def fast_nondominated_sort(F):
    """快速非支配排序"""
    S = [[] for _ in range(F.shape[0])]
    n = np.zeros(F.shape[0], dtype=int)
    fronts = [[]]
    
    for p_ in range(F.shape[0]):
        for q_ in range(F.shape[0]):
            if p_ == q_:
                continue
            if dominates(F[p_], F[q_]):
                S[p_].append(q_)
            elif dominates(F[q_], F[p_]):
                n[p_] += 1
        if n[p_] == 0:
            fronts[0].append(p_)
    
    i_ = 0
    while len(fronts[i_]) > 0:
        next_front = []
        for p_ in fronts[i_]:
            for q_ in S[p_]:
                n[q_] -= 1
                if n[q_] == 0:
                    next_front.append(q_)
        i_ += 1
        fronts.append(next_front)
    
    fronts.pop()
    return fronts


def crowding_distance(F, idxs):
    """拥挤距离计算"""
    if len(idxs) == 0:
        return {}
    
    m = F.shape[1]
    dist = {i: 0.0 for i in idxs}
    
    for j in range(m):
        vals = [(i, F[i, j]) for i in idxs]
        vals.sort(key=lambda x: x[1])
        dist[vals[0][0]] = float('inf')
        dist[vals[-1][0]] = float('inf')
        vmin = vals[0][1]
        vmax = vals[-1][1]
        denom = vmax - vmin if vmax > vmin else 1e-12
        for k_ in range(1, len(vals) - 1):
            prev_v = vals[k_-1][1]
            next_v = vals[k_+1][1]
            dist[vals[k_][0]] += (next_v - prev_v) / denom
    
    return dist


def nondominated_indices(F):
    """
    标准非支配排序算法（NSGA-II Fast Non-dominated Sort）
    返回rank=1（第一层）的非支配解索引
    
    参数：
        F: 目标值矩阵，形状为(M, nobj)，M为解的数量
    
    返回：
        rank=1的解的索引数组
    """
    M = F.shape[0]
    n = np.zeros(M, dtype=int)  # n[i]: 支配解i的解的数量
    S = [[] for _ in range(M)]  # S[i]: 被解i支配的解的列表
    
    # 计算支配关系
    for i in range(M):
        for j in range(i + 1, M):
            # 检查 i 是否支配 j
            if dominates(F[i], F[j]):
                S[i].append(j)
                n[j] += 1
            # 检查 j 是否支配 i
            elif dominates(F[j], F[i]):
                S[j].append(i)
                n[i] += 1
    
    # 第一层（rank=1）：n[i]=0的解
    front_1 = np.where(n == 0)[0]
    return front_1


def dominates(f1, f2):
    """
    ind1 支配 ind2 的条件：
      - 在所有目标上 ind1 <= ind2
      - 且至少一个目标上 ind1 < ind2
    这里只实现“f1 是否支配 f2”，不区分被谁支配的情况。
    """
    better = 0
    wbetter = 0
    nobj = len(f1)

    for i in range(nobj):
     
        if f1[i] <= f2[i] :
            better += 1
            
            if f1[i] < f2[i] :
                wbetter += 1
    
    return (better == nobj) and (wbetter >= 1)

def simple_nsga2_select(evolved_new_decs, evolved_new_objs, N):
    """
    完整的NSGA-II选择（非支配排序 + 拥挤距离）
    
    Args:
        evolved_new_decs: 候选个体决策变量
        evolved_new_objs: 候选个体目标值
        N: 需要选择的个体数量
        
    Returns:
        tuple: (选择后的决策变量, 选择后的目标值)
    """
    if evolved_new_decs.shape[0] <= N:
        return evolved_new_decs, evolved_new_objs
    
    
    # 1. 快速非支配排序
    fronts = fast_nondominated_sort(evolved_new_objs)
    
    selected_indices = []
    
    # 2. 按层选择
    for front in fronts:
        if len(selected_indices) + len(front) <= N:
            # 整层都可以选择
            selected_indices.extend(front)
        else:
            # 需要在当前层中选择部分个体
            need = N - len(selected_indices)
            if need > 0:
                # 计算拥挤距离
                crowding_dist = crowding_distance(evolved_new_objs, front)
                
                # 按拥挤距离降序排序
                front_sorted = sorted(front, key=lambda x: crowding_dist[x], reverse=True)

                selected_indices.extend(front_sorted[:need])
            break
    
    # 3. 返回选择的个体
    selected_indices = np.array(selected_indices[:N])
    return evolved_new_decs[selected_indices], evolved_new_objs[selected_indices]
