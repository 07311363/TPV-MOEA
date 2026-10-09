from parameter import Pro,rank,rank_a,R1,run
import copy
import statsmodels.tsa.stattools as st
import warnings
import os
path = os.getcwd()
output = os.path.join(path, 'PF')
warnings.filterwarnings("ignore")
os.makedirs(output, exist_ok=True)
Pro = Pro()
PRO2 = ['DF1','DF2','DF3','DF4','DF5','DF6','DF7','DF8','DF9']
#PRO2 = ['DF1','DF7']
#PRO2 = ['FDA1','FDA2','FDA3']
PRO0 = ['FDA5']
# 排序模式可选项：
#   "gcn_dijkstra"     无监督GCN + Dijkstra
sort_mode = "gcn_dijkstra"
# 参考点绑定模式："scaled_weight"（默认）或 "euclidean"（欧氏距离绑定）或 "no_weight_scale"
ref_bind_mode = "scaled_weight"

from gcn_sort.inference import rank_with_gcn
from response_strategy import (
    detect_shared_points,
    build_regions,
    predict_population_dual_strategy,
    predict_population_with_regions,
    nondominated_indices,
)

for pro_str in PRO2:#修改问题，二维和三维需分开跑
    Pro.set_pro(pro_str)
    import numpy as np
    from numpy import floor
    from test_problem import Problem
    from parameter import N,Gen,taot,nvar,p,T,T0,nobj
    from moead import MOEAD
    from jMetalPy.jmetal.core.quality_indicator import HyperVolume
    import pickle
    from tqdm import tqdm
    import time

    start =time.time()

    HyperVolume = HyperVolume(Pro.hv_z)
    for number in range(1,run+1):
        print('运算次数:', number)
        I = int(Gen / taot)
        moead = MOEAD(pro_str)
        low_bounds = np.empty(nvar, dtype=np.float64)
        upp_bounds = np.empty(nvar, dtype=np.float64)
        low_bounds[:moead.lu] = moead.Lb1
        upp_bounds[:moead.lu] = moead.Ub1
        low_bounds[moead.lu:] = moead.Lb2
        upp_bounds[moead.lu:] = moead.Ub2
        pro = Problem(pro_str)
        weight = moead.weight_vector()
        neighbor = moead.neighbor(weight)
        sol,fitness = moead.initial_population()
        z = moead.initial_z(fitness)  # 初始化理想点
        Fit = np.zeros((N,nobj,int((Gen-T0) / taot)+1))
        all_s = np.zeros((N,nvar,int((Gen-T0)/ taot)+1))
        # 初始化历史变量（用于响应策略）
        center_history = []  # 记录中心点
        dt_history = []
        prev_sol = None
        prev_fitness = None
        prev_nd_dec = None
        prev_nd_obj = None
        prev_center = None
        # 边界变量（用于局部随机策略）
        oldmin = None
        oldmax = None
        newmin = None
        newmax = None
        memory_nondominated = []  # 存储历史各环境的ND解

        for gen in tqdm(range(1,Gen)):
            tau_tmp = max(gen - (T0 + 1), 0)
            K = int(floor(tau_tmp / taot))
            if gen >= taot and (gen-T0)%taot == 1:
                # =================== 环境变化检测 ===================
                # 保存数据到历史记录
                Fit[0:N, 0:nobj, K - 1] = fitness
                all_s[:, :, K - 1] = sol
                nd_idx = nondominated_indices(Fit[:, :, K - 1])
                nd_dec = all_s[nd_idx, :, K - 1]
                current_center = np.mean(nd_dec, axis=0) if nd_dec.shape[0] > 0 else np.zeros(nvar)
                center_history.append(current_center.copy())
                nd_obj = np.array(
                    [moead.problem(x, gen) for x in nd_dec]
                ) if nd_dec.shape[0] > 0 else np.empty((0, nobj))
                # =================== 响应策略开始 ===================
                if K == 1:
                    oldmin = 1e10
                    oldmax = -1e10
                    newmin = np.min(nd_dec, axis=0)
                    newmax = np.max(nd_dec, axis=0)
                    portion = 0.2

                    n_random = int(N * portion)
                    all_indices = np.arange(N)
                    random_idx = np.random.choice(all_indices, n_random, replace=False)

                    for idx in random_idx:
                        sol[idx] = np.random.rand(nvar)
                        fitness[idx] = moead.problem(sol[idx], gen)

                    fitness = np.array([moead.problem(s, gen) for s in sol])

                elif K >= 2:
                    prev_center = center_history[-2] if len(center_history) >= 2 else np.zeros(nvar)
                    Dt_raw = current_center - prev_center
                    Dt = np.abs(Dt_raw)
                    Dt_norm = np.linalg.norm(Dt)
                    dt_history.append(Dt.copy())

                    if len(dt_history) > 0 and len(Dt) > 0:
                        sigma = np.mean(dt_history, axis=0)
                    else:
                        sigma = np.zeros(nvar)

                    prev_nd_dec = memory_nondominated[-1][0] if len(memory_nondominated) > 0 else nd_dec
                    v_prime_dec = prev_nd_dec + Dt_raw
                    # 边界检查：平移后的v_prime_dec需要检查是否超出全局决策空间
                    from response_strategy import boundary_check
                    for i in range(v_prime_dec.shape[0]):
                        v_prime_dec[i] = boundary_check(v_prime_dec[i], low_bounds, upp_bounds)
                    # v_prime_obj
                    v_prime_obj = np.array([moead.problem(vp, gen) for vp in v_prime_dec]) if v_prime_dec.size > 0 else v_prime_dec

                    # 当前时刻的ND解
                    new_nd_dec = nd_dec
                    new_nd_obj = nd_obj
                    oldmin = newmin
                    oldmax = newmax
                    newmin = np.min(new_nd_dec, axis=0)
                    newmax = np.max(new_nd_dec, axis=0)
                    if v_prime_obj.size > 0 and new_nd_obj.size > 0:
                        # 检测共享点
                        boundary_idx, shared_idx = detect_shared_points(v_prime_obj, new_nd_obj, nobj)
                        shared_all_idx = np.array(sorted(list(set(boundary_idx + shared_idx))), dtype=int)
                        # 无监督GCN邻域聚合 + Dijkstra全局排序（无需模型、无需标签）
                        order, s_scores = rank_with_gcn(
                                v_prime_obj,
                                shared_all_idx,
                                cfg={
                                    'k': 7,
                                    'layers': 2,
                                    'sort_mode': sort_mode
                                }
                        ) if shared_all_idx.size > 0 else (np.array([], dtype=int), np.array([]))
                        shared_idx_sorted = shared_all_idx[order] if shared_all_idx.size > 0 else shared_all_idx

                        prev_pop_dec = all_s[:, :, K - 1]
                        regions = build_regions(shared_idx_sorted, v_prime_obj, v_prime_dec, new_nd_obj, nobj,
                                              Dt_raw, prev_pop_dec, nvar, moead, gen,
                                              low_bounds=low_bounds, upp_bounds=upp_bounds,
                                              ref_bind_mode=ref_bind_mode)

                        sol, fitness = predict_population_dual_strategy(
                                regions, sol, fitness, Dt_raw, moead, gen,
                                nd_dec=new_nd_dec, nd_obj=new_nd_obj, sigma=sigma,
                                oldmin=oldmin, oldmax=oldmax, newmin=newmin, newmax=newmax,
                                low_bounds=low_bounds, upp_bounds=upp_bounds )

                    fitness = np.array([moead.problem(s, gen) for s in sol])
                # 存储当前时刻非支配解集
                memory_nondominated.append((nd_dec.copy(), nd_obj.copy()))
                # =================== 响应策略结束 ===================
            # =================== 静态MOEAD优化 ===================
            for i in range(N):
                if np.random.rand() < p:
                    P = neighbor[i, 0:T]
                else:
                    P = list(range(N))

                V, Fitness = moead.reproduction(P, sol, i, gen)
                z = moead.update_z(z, Fitness)
                sol, fitness = moead.update_population(P, sol, weight, z, V, fitness, Fitness)
            time.sleep(0.01)  # 进度条更新

        # 存储当前代的fitness和sol到历史记录
        Fit[:, :, K] = fitness
        data = Fit
        for tt in range(0, int((Gen-T0)/taot) ):
            # 提取第tt个环境的整个种群
            current_fitness = data[:, :, tt]
            
            # 保存整个种群（所有N个个体）
            files = output + '\\pf_' + str(pro_str)+ '_' + str(number)+ '_' + str(tt) + '.dat'
            f = open(files, "w", encoding='utf-8')
            for j in range(current_fitness.shape[0]):
                if nobj==2:
                    f.write(str(current_fitness[j, 0]) + '  ' + str(current_fitness[j, 1]))
                    f.write('\n')
                else:
                    f.write(str(current_fitness[j, 0]) + '  ' + str(current_fitness[j, 1])+ '  ' + str(current_fitness[j, 2]))
                    f.write('\n')
            f.close()
        
        # 关闭本次运行的日志
        # logger.close()  # logger未定义，注释掉
        
        # 关闭共享点调试器
        # close_shared_points_debugger()  # 函数未定义，注释掉

        end = time.time()
        print('Running time: %s Seconds' % (end - start))
