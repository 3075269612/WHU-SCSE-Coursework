"""
2.3.2 传教士与野人问题 (Missionaries and Cannibals)
=====================================================
分别使用 DFS 和 BFS 求解经典的传教士与野人过河问题。

问题描述:
  3个传教士和3个野人在河的右岸，要用一条船全部运到左岸。
  约束条件:
    1. 船每次最多载2人（至少1人划船）
    2. 在任意一岸，如果传教士人数 > 0，则传教士人数必须 >= 野人人数
       （否则传教士会被吃掉）

状态表示 (M1, C1, B, M2, C2):
  M1: 左岸传教士数
  C1: 左岸野人数
  B:  船的位置 (1=右岸, 0=左岸)
  M2: 右岸传教士数
  C2: 右岸野人数

  初始状态: (0, 0, 1, 3, 3)  —— 全部在右岸，船在右岸
  目标状态: (3, 3, 0, 0, 0)  —— 全部在左岸，船在左岸
"""

from collections import deque


def is_valid(state):
    """检查状态是否合法。

    两个条件:
      1. 所有人数必须非负
      2. 任一岸上，如果没有传教士则不受限制；
         如果有传教士，则传教士人数不能少于野人人数
    """
    M1, C1, _, M2, C2 = state
    if M1 >= 0 and M2 >= 0 and C1 >= 0 and C2 >= 0:
        if (M1 == 0 or M1 >= C1) and (M2 == 0 or M2 >= C2):
            return True
    return False


# ==================== 1) 深度优先搜索 (DFS) ====================

def dfs(state, goal, visited, path):
    """DFS递归搜索——沿着一条分支深入到底，找不到再回溯试另一分支。

    参数:
        state:   当前状态
        goal:    目标状态
        visited: 已访问状态集合（防止重复、避免回路）
        path:    从起点到当前状态的路径列表

    返回:
        完整的路径列表，未找到返回 None

    注意: DFS 不保证找到的是最短路径（但在本问题中恰好最短）。
    """
    if state == goal:
        return path

    visited.add(state)

    M1, C1, B, M2, C2 = state

    # 根据船的位置生成5种可能的划船方案:
    #   (1个传教士, 2个传教士, 1个野人, 2个野人, 1传1野)
    moves = []
    if B == 1:  # 船在右岸 — 人员从右岸 → 左岸 (左岸增加, 右岸减少)
        moves = [
            (M1 + 1, C1, 0, M2 - 1, C2),           # 送 1 传教士
            (M1 + 2, C1, 0, M2 - 2, C2),           # 送 2 传教士
            (M1, C1 + 1, 0, M2, C2 - 1),           # 送 1 野人
            (M1, C1 + 2, 0, M2, C2 - 2),           # 送 2 野人
            (M1 + 1, C1 + 1, 0, M2 - 1, C2 - 1),   # 送 1 传+1 野
        ]
    else:  # 船在左岸 — 人员从左岸 → 右岸 (左岸减少, 右岸增加)
        moves = [
            (M1 - 1, C1, 1, M2 + 1, C2),           # 送回 1 传教士
            (M1 - 2, C1, 1, M2 + 2, C2),           # 送回 2 传教士
            (M1, C1 - 1, 1, M2, C2 + 1),           # 送回 1 野人
            (M1, C1 - 2, 1, M2, C2 + 2),           # 送回 2 野人
            (M1 - 1, C1 - 1, 1, M2 + 1, C2 + 1),   # 送回 1 传+1 野
        ]

    for new_state in moves:
        if is_valid(new_state) and new_state not in visited:
            result = dfs(new_state, goal, visited, path + [new_state])
            if result:
                return result  # 深度优先：找到就立即返回
    return None


def solve_missionaries_dfs(start_state, goal_state):
    """使用DFS求解传教士与野人问题"""
    solution = dfs(start_state, goal_state, set(), [start_state])

    if solution:
        print("=== DFS 搜索结果 ===")
        count = 0
        for step in solution:
            count = count + 1
            print(step)
        print("Steps:", count - 1)
    else:
        print("Fail to find a solution!")
    print()


# ==================== 2) 广度优先搜索 (BFS) ====================

def bfs(start, goal):
    """BFS搜索——逐层扩展，保证第一步找到的解就是最短路径。

    参数:
        start: 初始状态
        goal:  目标状态

    返回:
        最短路径的状态列表

    核心:
        deque 作为 FIFO 队列，每个节点附带它走过的完整路径。
        第N层的所有节点遍历完才进入N+1层。
        visited 在入队时即标记，避免同一状态重复入队。
    """
    queue = deque([(start, [start])])  # (当前状态, 从起点到此状态的路径)
    visited = set()

    while queue:
        state, path = queue.popleft()

        if state == goal:
            return path

        visited.add(state)

        M1, C1, B, M2, C2 = state
        moves = []
        if B == 1:  # 船在右岸
            moves = [
                (M1 + 1, C1, 0, M2 - 1, C2),
                (M1 + 2, C1, 0, M2 - 2, C2),
                (M1, C1 + 1, 0, M2, C2 - 1),
                (M1, C1 + 2, 0, M2, C2 - 2),
                (M1 + 1, C1 + 1, 0, M2 - 1, C2 - 1)
            ]
        else:  # 船在左岸
            moves = [
                (M1 - 1, C1, 1, M2 + 1, C2),
                (M1 - 2, C1, 1, M2 + 2, C2),
                (M1, C1 - 1, 1, M2, C2 + 1),
                (M1, C1 - 2, 1, M2, C2 + 2),
                (M1 - 1, C1 - 1, 1, M2 + 1, C2 + 1)
            ]

        for new_state in moves:
            if is_valid(new_state) and new_state not in visited:
                queue.append((new_state, path + [new_state]))
    return None


def solve_missionaries_bfs(start_state, goal_state):
    """使用BFS求解传教士与野人问题"""
    solution = bfs(start_state, goal_state)

    if solution:
        print("=== BFS 搜索结果 ===")
        count = 0
        for step in solution:
            count = count + 1
            print(step)
        print("Steps:", count - 1)
    else:
        print("Fail to find a solution!")
    print()


# ==================== 主程序 ====================
if __name__ == "__main__":
    start_state = (0, 0, 1, 3, 3)  # 初始: 右岸3传教士, 3野人, 船在右岸
    goal_state = (3, 3, 0, 0, 0)   # 目标: 左岸3传教士, 3野人, 船在左岸

    print("=" * 50)
    print("2.3.2 传教士与野人问题")
    print("=" * 50)
    print()

    solve_missionaries_dfs(start_state, goal_state)
    solve_missionaries_bfs(start_state, goal_state)
