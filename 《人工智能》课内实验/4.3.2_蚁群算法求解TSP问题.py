# 4.3.2 蚁群算法案例 - 解决旅行商问题(TSP)
import random
import numpy as np

# 定义城市坐标
city_location = {
    'A': (5, 10), 'B': (6, 15), 'C': (10, 15), 'D': (14, 14), 'E': (20, 10),
    'F': (16, 5), 'G': (8, 5), 'H': (4, 8), 'I': (8, 12), 'J': (12, 12)
}

class Graph(object):
    """图类，用于存储城市的距离以及信息素浓度等环境信息"""
    def __init__(self, city_location, pheromone=1.0, alpha=1.0, beta=1.0):
        self.city_location = city_location
        self.n = len(city_location)
        # 初始化 pheromone(信息素) 矩阵
        self.pheromone = [[pheromone for _ in range(self.n)] for _ in range(self.n)]
        self.alpha = alpha
        self.beta = beta
        # 生成城市之间的欧氏距离矩阵
        self.distances = self._generate_distances()

    def _generate_distances(self):
        distances = []
        city_keys = list(self.city_location.keys())
        for index_i, _i in enumerate(city_keys):
            row = []
            for index_j, _j in enumerate(city_keys):
                if _i == _j:
                    row.append(0)
                elif _i < _j:
                    x1, y1 = self.city_location[_i]
                    x2, y2 = self.city_location[_j]
                    row.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
                else:
                    # 补齐距离矩阵斜对称位置
                    row.append(distances[index_j][index_i])
            distances.append(row)
        return distances

class Ant(object):
    """蚂蚁个体类"""
    def __init__(self, graph, start_city):
        self.graph = graph
        self.start_city = start_city
        self.curr_city = start_city
        
        # 记录访问情况，并将初始城市标记为已访问
        self.visited = [False for _ in range(graph.n)]
        self.visited[start_city] = True
        
        self.tour_length = 0
        self.tour = [start_city]

    def choose_next_city(self):
        """基于信息素和启发信息(距离倒数)，用轮盘赌来选择下一个城市"""
        probs = []
        total_prob = 0
        
        # 遍历所有周边城市
        for i in range(self.graph.n):
            if not self.visited[i]:
                # 计算前往未访问城市的概率因子
                prob = (self.graph.pheromone[self.curr_city][i] ** self.graph.alpha * 
                        (1.0 / self.graph.distances[self.curr_city][i]) ** self.graph.beta)
                probs.append((i, prob))
                total_prob += prob

        # 利用随机数进行轮盘赌选择
        r = random.uniform(0, total_prob)
        upto = 0
        for i, prob in probs:
            if upto + prob >= r:
                # 更新状态
                self.curr_city = i
                self.visited[i] = True
                self.tour.append(i)
                self.tour_length += self.graph.distances[self.tour[-2]][i]
                return
            upto += prob

    def tour_complete(self):
        """是否已走完所有城市"""
        return len(self.tour) == self.graph.n

def ant_colony(graph, num_ants, num_iterations, evaporation_rate=0.5, q=500):
    """
    蚁群算法核心外层调度
    """
    shortest_tour = None
    shortest_tour_length = float('inf')
    
    for _ in range(num_iterations):
        # 随机生成蚂蚁，放置在不同起点
        ants = [Ant(graph, random.randint(0, graph.n - 1)) for _ in range(num_ants)]
        
        for ant in ants:
            # 单个蚂蚁完成一整圈巡游
            while not ant.tour_complete():
                ant.choose_next_city()
            
            # 加入回到起始点的最后一程距离，使路线闭合
            ant.tour_length += graph.distances[ant.tour[-1]][ant.start_city]
            
            # 记录最短路径及其长度
            if ant.tour_length < shortest_tour_length:
                shortest_tour_length = ant.tour_length
                shortest_tour = ant.tour[:]
                
        # 更新全图的信息素浓度
        # 1. 蒸发：随时间逐步消退残余信息素
        for u in range(graph.n):
            for v in range(graph.n):
                graph.pheromone[u][v] = (1.0 - evaporation_rate) * graph.pheromone[u][v]
                
        # 2. 沉积：优良路径由于其 tour_length 更小，沉积的信息素更多
        for ant in ants:
            for j in range(graph.n - 1):
                graph.pheromone[ant.tour[j]][ant.tour[j + 1]] += q / ant.tour_length
            graph.pheromone[ant.tour[-1]][ant.start_city] += q / ant.tour_length
            
    return shortest_tour, shortest_tour_length

def main():
    graph = Graph(city_location)
    num_ants = 20
    num_iterations = 100
    
    shortest_tour, shortest_tour_length = ant_colony(graph, num_ants, num_iterations)
    
    # 路径下标重映射回城市名称
    city_keys = list(city_location.keys())
    best_path = [city_keys[i] for i in shortest_tour]
    
    print("最短路径：", best_path)
    print("最短距离：", shortest_tour_length)

if __name__ == '__main__':
    main()