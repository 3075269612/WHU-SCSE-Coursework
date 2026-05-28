# 4.3.3 模拟退火算法案例 - 解决旅行商问题(TSP)
from typing import Dict, List, Tuple
import numpy as np

def read_dataset():
    """定义含有 20 个城市坐标的坐标字典作为测试数据集。"""
    city_coordinates_ = {
        0: (60, 200), 1: (180, 200), 2: (80, 180), 3: (140, 180),
        4: (20, 160), 5: (100, 160), 6: (200, 160), 7: (140, 140),
        8: (40, 120), 9: (100, 120), 10: (180, 100), 11: (60, 80),
        12: (120, 80), 13: (180, 60), 14: (20, 40), 15: (100, 40),
        16: (200, 40), 17: (20, 20), 18: (60, 20), 19: (160, 20)
    }
    return city_coordinates_

def get_distance_matrix(city_coordinates: Dict[int, Tuple[int, int]], num_city: int) -> np.ndarray:
    """计算出各个城市之间的欧氏距离矩阵。"""
    distance_matrix = np.empty(shape=(num_city, num_city), dtype=np.float64)
    for i in range(num_city):
        xi, yi = city_coordinates[i]
        for j in range(num_city):
            xj, yj = city_coordinates[j]
            distance_matrix[i][j] = np.sqrt(np.power(xi - xj, 2) + np.power(yi - yj, 2))
    return distance_matrix

def eval_func(x: np.ndarray, distance_matrix: np.ndarray, num_city: int):
    """
    评价函数，计算整个循环路径的总距离
    :param x: 含有旅行路径数字标记(0-N)的解向量
    :return: 解的目标函数值 (路径总距离)
    """
    total_distance = 0
    # 累加沿途各段位
    for k in range(num_city - 1):
        total_distance += distance_matrix[x[k]][x[k + 1]]
    # 回到终点以闭合路径
    total_distance += distance_matrix[x[-1]][x[0]]
    return total_distance

def two_opt(x: np.ndarray, num_city: int):
    """
    2-opt swap，通过将序列中两个体位置对调扰动生成新解用于评估。
    :param x: 当前解向量
    :return: 产生的新解
    """
    x_new = x.copy()
    r1 = np.random.randint(low=0, high=num_city)
    r2 = np.random.randint(low=0, high=num_city)
    x_new[r1], x_new[r2] = x_new[r2], x_new[r1]
    return x_new

def run_simulated_annealing(distance_matrix: np.ndarray, num_city: int, 
                            temp_initial=1e6, temp_final=0.1, alpha=0.98) -> List:
    """
    控制模拟退火迭代优化全流程主函数。
    :param temp_initial: 初始温度 T0
    :param temp_final: 终止温度 Tf
    :param alpha: 每次外循环时的降温系数
    :return: 每一代最优解及解的目标函数值构成的时间追踪线 trace 列表
    """
    temp_current = temp_initial  # 预置当前温度
    
    # 产生初始解编码：随机产生1~N 的排列
    x = np.random.permutation(num_city)  
    obj_value = eval_func(x, distance_matrix, num_city)
    
    global_best = x.copy()  # 初始化当前全局最优解
    trace: List[Tuple[np.ndarray, float]] = [(x, obj_value)]  # 用于记录变化轨迹
    
    # 外循环: 退火全局过程操作
    while temp_current > temp_final:  
        
        # 内循环: 马尔可夫链 (在等温条件下进行足够的扰动寻优)
        for i in range(1000):  
            obj_value_old = eval_func(x, distance_matrix, num_city)
            
            # 使用 2-opt 进行邻域扰动搜索
            x_ = two_opt(x, num_city)
            obj_value_new = eval_func(x_, distance_matrix, num_city)
            
            # 计算距离差分(若目标变大，其值为正 -> 退化解；目标缩小，其值为负 -> 优良解)
            delta_temp = obj_value_new - obj_value_old
            
            # 若新解引起目标函数变差（路线总距离增大）
            if delta_temp > 0:
                # 按照依据温差与当前总温度拟合出来的 Metrolopis 接收准则决定是否概率接受该“恶化”解，从而帮助跳出局部最优区
                if np.random.uniform(0, 1) < 1.0 / np.exp(delta_temp / temp_current):
                    x = x_
                    
            # 找到优良解，则百分百接纳该新变移情况
            else:
                x = x_
                global_best = x.copy()
                
        # 保存这个温度世代演化后的最佳方案
        trace.append((global_best, eval_func(global_best, distance_matrix, num_city)))
        
        # 以固定的指数步调对系统下温
        temp_current *= alpha  
        
    return trace

def print_solution(trace: List[Tuple[np.ndarray, float]], num_city: int) -> None:
    """对比打印初始与末尾世代的最优解及成绩。"""
    print(f'城市数量: {num_city}')
    initial_solution, initial_obj_value = trace[0]
    final_solution, final_obj_value = trace[-1]
    print(f'最初路径: {list(initial_solution)}, 最初路径总距离: {initial_obj_value}')
    print(f'最终路径: {list(final_solution)}, 最终路径总距离: {final_obj_value}')

def main():
    city_coordinates = read_dataset()
    num_city = len(city_coordinates)
    distance_matrix = get_distance_matrix(city_coordinates, num_city)
    
    # 进行模拟退火过程，获得沿途路径优化轨迹
    trace = run_simulated_annealing(distance_matrix, num_city)
    print_solution(trace, num_city)

if __name__ == "__main__":
    main()