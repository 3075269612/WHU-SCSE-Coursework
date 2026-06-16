"""
2.3.6 旅行商问题 (Traveling Salesman Problem, TSP)
=====================================================
使用 A* 启发式搜索 + 分支限界剪枝求解小规模 TSP 问题。

问题描述:
  有 n 个城市，已知任意两城市之间的距离。
  从城市 A 出发，访问所有其他城市各一次，最后返回 A，
  求总路程最短的访问顺序（即最短哈密顿回路）。

  本例: 4个城市 A, B, C, D (索引 0,1,2,3)
  距离矩阵:
        A   B   C   D
    A   0  10  15  20
    B  10   0  35  25
    C  15  35   0  30
    D  20  25  30   0

  最优解: A → B (10) → D (25) → C (30) → A (15) = 80

算法思路:
  —— 将 TSP 建模为状态空间搜索 ——
  状态 (cost, city, path): 已花费 cost, 当前在 city, 已访问路径 path
  动作: 选择下一个未访问的城市
  目标: 访问完所有城市后回到起点

  —— A* + 分支限界 ——
  用优先队列按 f = g + h 排序:
    g = cost: 已走过的累积距离
    h = mst_weight(剩余未访问节点 ∪ {当前城市, 起点}): 最小生成树下界
      剩余路径(当前→未访问各城→起点) 是一条哈密顿路径，
      而哈密顿路径是一棵生成树的子集，
      因此 MST 权重 ≤ 实际剩余代价，满足可采纳性(不高估)。

  分支限界: 若 f >= best_cost，该分支不可能更优，直接跳过(剪枝)。
"""

import heapq


def mst_weight(graph, nodes):
    """Prim 算法: 计算节点集合 nodes 上最小生成树(MST)的权重。

    参数:
        graph: 完整的距离矩阵
        nodes: 要计算 MST 的节点索引列表

    返回:
        MST 的总权重（作为剩余路径代价的下界）

    算法步骤:
      1. 从 nodes[0] 开始，将其加入 visited
      2. 每次从 visited 到 unvisited 中选一条最小边，
         把对应的未访问节点加入 visited，累加边权
      3. 重复直到所有节点连通

    为何可纳:
      剩余待完成的部分是: 当前城市 → (若干未访问城市) → 起点
      这是一条路径，去掉其中某条边后它是一棵生成树，
      因此 MST 权重不会超过这条路径的实际长度。
    """
    if len(nodes) <= 1:
        return 0

    visited = {nodes[0]}
    unvisited = set(nodes[1:])
    total = 0

    while unvisited:
        min_edge = float('inf')
        best = None
        # 在 visited 和 unvisited 之间找最小边
        for u in visited:
            for v in unvisited:
                if graph[u][v] < min_edge:
                    min_edge = graph[u][v]
                    best = v
        visited.add(best)
        unvisited.remove(best)
        total += min_edge

    return total


def a_star_tsp(graph):
    """A* 搜索 + 分支限界求解 TSP。

    队列元素: (f, cost, city, path)
      f    = cost + h (总估计代价)
      cost = 已走距离 (g)
      city = 当前所在城市
      path = 已访问城市序列

    剪枝策略: 当 f >= best_cost 时剪枝，因为不可能更优。
    """
    n = len(graph)
    start = 0  # 从城市 A(索引0) 出发

    open_set = [(0, 0, start, [start])]  # (f, cost, city, path)
    best_cost = float('inf')
    best_path = None

    while open_set:
        f, cost, city, path = heapq.heappop(open_set)

        # 分支限界: 当前分支的最优估计已不优于已知最优，剪枝
        if f >= best_cost:
            continue

        # 所有城市已访问 — 补上回到起点的距离形成完整回路
        if len(path) == n:
            total = cost + graph[city][start]
            if total < best_cost:
                best_cost = total
                best_path = path + [start]
            continue

        # 扩展到每个未访问的城市
        for nxt in range(n):
            if nxt not in path:
                new_cost = cost + graph[city][nxt]
                new_path = path + [nxt]

                # 剩余节点集合: 起点 + 当前城市 + 所有未访问城市
                # MST 在这些节点上计算，作为启发式下界
                h_nodes = [nxt, start] + [i for i in range(n) if i not in new_path]
                h = mst_weight(graph, h_nodes)
                new_f = new_cost + h

                # 分支限界: 只将 f < best_cost 的节点入队
                if new_f < best_cost:
                    heapq.heappush(open_set, (new_f, new_cost, nxt, new_path))

    return best_cost, best_path


if __name__ == "__main__":
    print("=" * 50)
    print("2.3.6 旅行商问题")
    print("=" * 50)
    print()

    # 距离矩阵: graph[i][j] = 城市 i 到城市 j 的距离
    graph = [
        [0, 10, 15, 20],   # A
        [10, 0, 35, 25],   # B
        [15, 35, 0, 30],   # C
        [20, 25, 30, 0],   # D
    ]

    city_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    min_cost, path = a_star_tsp(graph)
    print("城市距离矩阵:")
    for i, row in enumerate(graph):
        print(f"  {city_names[i]}: {row}")
    print()
    print("最小成本:", min_cost)
    print("旅行路径:", path, "即", "->".join(city_names[p] for p in path))
