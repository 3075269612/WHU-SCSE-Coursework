"""
2.3.1 八数码问题 (8-Puzzle)
===============================
分别使用 DFS、BFS、A* 三种搜索算法求解八数码问题。

问题描述:
  3×3 棋盘上有 1~8 共八个数字和一个空白格(用0表示)。
  每次只能将空白格与相邻(上下左右)的数字交换位置。
  从初始状态出发，找到到达目标状态的最少移动步数。

状态表示:
  一维列表/元组，长度为9，索引 0~8 对应棋盘从左到右、从上到下:
    索引: 0 1 2
         3 4 5
         6 7 8

  初始状态: [1, 4, 2, 8, 6, 3, 7, 0, 5]
    对应棋盘:  1 4 2
              8 6 3
              7 _ 5
  目标状态: [1, 2, 3, 8, 0, 4, 7, 6, 5]
    对应棋盘:  1 2 3
              8 _ 4
              7 6 5
"""

from collections import deque
import heapq


# ==================== 状态移动 ====================

def move(state, action):
    """根据动作移动空白格(0)，返回新状态。

    参数:
        state: 当前状态列表，如 [1,4,2,8,6,3,7,0,5]
        action: 移动方向，取 'up'|'down'|'left'|'right'

    返回:
        移动后的新状态列表（原地交换空白格与相邻数字）

    合法性检查:
        - up:    空白格不在第一行 (索引不是 0,1,2)
        - down:  空白格不在第三行 (索引不是 6,7,8)
        - left:  空白格不在第一列 (索引不是 0,3,6)
        - right: 空白格不在第三列 (索引不是 2,5,8)
    """
    next_state = state[:]
    index = next_state.index(0)  # 找到空白格位置

    if action == 'up':
        if index not in [0, 1, 2]:
            next_state[index - 3], next_state[index] = next_state[index], next_state[index - 3]
    elif action == 'down':
        if index not in [6, 7, 8]:
            next_state[index + 3], next_state[index] = next_state[index], next_state[index + 3]
    elif action == 'left':
        if index not in [0, 3, 6]:
            next_state[index - 1], next_state[index] = next_state[index], next_state[index - 1]
    elif action == 'right':
        if index not in [2, 5, 8]:
            next_state[index + 1], next_state[index] = next_state[index], next_state[index + 1]
    return next_state


# ==================== 1) 深度优先搜索 (DFS) ====================

def dfs(state, path, max_depth, goal_state):
    """DFS递归搜索——沿着一条路径深入到底，找不到再回溯。

    参数:
        state:     当前状态
        path:      从起点到当前状态的状态序列（同时充当 visited 集合）
        max_depth: 最大搜索深度（防止无限递归）
        goal_state:目标状态

    返回:
        找到的完整路径（状态列表），未找到返回 None
    """
    if state == goal_state:
        return path
    if len(path) >= max_depth:
        return None

    for action in ['up', 'down', 'left', 'right']:
        next_state = move(state, action)
        if next_state not in path:  # 避免回路：path 本身即为访问记录
            result = dfs(next_state, path + [next_state], max_depth, goal_state)
            if result is not None:
                return result  # 找到即返回，不保证最优
    return None


def solve_8puzzle_dfs(start_state, goal_state):
    """使用迭代加深DFS求解八数码问题。

    从深度=1开始，逐步增加深度限制，直到找到解或达到上限30。
    这样做的好处是：
      1. 避免DFS在无限分支中永远不返回
      2. 在深度较浅时更快找到解
    """
    result = None
    for depth in range(1, 31):
        result = dfs(start_state, [start_state], depth, goal_state)
        if result is not None:
            break

    if result is None:
        print("Fail to find a path within 30 steps!")
    else:
        print("=== DFS 搜索结果 ===")
        print("start_state =", start_state)
        for item in result[1:-1]:
            print("next_state =", item)
        print("goal_state =", goal_state)
        print("Steps:", len(result) - 1)
    print()


# ==================== 2) 广度优先搜索 (BFS) ====================

def bfs(start, goal_state, max_steps):
    """BFS搜索——逐层扩展，保证找到的路径步数最少。

    参数:
        start:      初始状态
        goal_state: 目标状态
        max_steps:  最大搜索步数

    返回:
        动作序列列表（如 ['up','left','down',...]），未找到返回 None

    核心:
        使用 deque 作为 FIFO 队列，每次从队首取出状态扩展，
        新状态追加到队尾，从而实现"先浅后深"的逐层搜索。
    """
    visited = set()
    queue = deque([(start, [])])  # (当前状态, 到达此状态的动作序列)

    while queue:
        state, path = queue.popleft()  # FIFO: 取出最先加入的
        if state == goal_state:
            return path
        if len(path) >= max_steps:
            continue

        for action in ['up', 'down', 'left', 'right']:
            next_state = move(state, action)
            if tuple(next_state) not in visited:
                visited.add(tuple(next_state))
                queue.append((next_state, path + [action]))
    return None


def solve_8puzzle_bfs(start_state, goal_state):
    """使用BFS求解八数码问题。将动作序列还原为状态序列后输出。"""
    result = bfs(start_state, goal_state, 30)

    if result is None:
        print("Fail to find a path within 30 steps!")
    else:
        # 从动作序列回放，重建完整的状态路径
        moves = [start_state]
        for item in result:
            next_state = move(moves[-1], item)
            moves.append(next_state)

        print("=== BFS 搜索结果 ===")
        print("start_state =", start_state)
        for item in moves[1:-1]:
            print("next_state =", item)
        print("goal_state =", goal_state)
        print("Steps:", len(result))
    print()


# ==================== 3) A* 算法 ====================

class PuzzleNode:
    """A*搜索的搜索节点。

    g: 从起点到当前状态的实际代价（走过的步数）
    h: 启发式估计值（到目标状态的估计剩余步数）
    f = g + h: 总估计代价，优先队列按此排序
    """
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state    # 棋盘状态（元组，可哈希）
        self.parent = parent  # 父节点，用于最终回溯路径
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        """小根堆比较：f 值小的优先弹出"""
        return self.f < other.f


def manhattan_distance(state, goal):
    """曼哈顿距离启发式——A*的核心。

    对每个数字 1~8，计算它在当前棋盘中位置与目标棋盘位置的
    |行差| + |列差|，然后求和。

    例如：数字 2 在 state[1] = (0,1)，在 goal[1] = (0,0):
      曼哈顿距离 = |0-0| + |1-0| = 1

    此启发式是可纳的(admissible)：永远不会高估实际代价，
    因此 A* 保证找到最优解。
    """
    distance = 0
    for i in range(1, 9):
        xi, yi = divmod(state.index(i), 3)  # 数字i在当前状态的行列
        xg, yg = divmod(goal.index(i), 3)   # 数字i在目标状态的行列
        distance += abs(xi - xg) + abs(yi - yg)
    return distance


def get_neighbors(state):
    """获取当前状态的合法邻居状态（所有可能的下一步空白格移动结果）。

    用行列坐标定位空白格，向上下左右四个方向尝试移动，
    过滤掉越界的移动，返回所有合法的新状态。
    """
    neighbors = []
    index = state.index(0)
    row, col = divmod(index, 3)
    moves = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    for r, c in moves:
        if 0 <= r < 3 and 0 <= c < 3:
            new_index = r * 3 + c
            new_state = list(state)
            new_state[index], new_state[new_index] = new_state[new_index], new_state[index]
            neighbors.append(tuple(new_state))
    return neighbors


def a_star(start, goal):
    """A*搜索算法。

    使用优先队列(小根堆)管理待探索节点，每次取出 f=g+h 最小的节点扩展。
    """
    open_list = []  # OPEN表：待扩展节点
    closed_set = set()  # CLOSED表：已扩展状态
    heapq.heappush(open_list, PuzzleNode(start, g=0, h=manhattan_distance(start, goal)))

    while open_list:
        current_node = heapq.heappop(open_list)  # 取 f 最小节点

        if current_node.state == goal:
            # 找到目标，通过 parent 回溯路径
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]  # 反转得到起点->终点的顺序

        closed_set.add(current_node.state)

        for neighbor in get_neighbors(current_node.state):
            if neighbor in closed_set:
                continue
            g = current_node.g + 1                      # 每走一步代价+1
            h = manhattan_distance(neighbor, goal)       # 启发式估计剩余步数
            neighbor_node = PuzzleNode(neighbor, current_node, g, h)
            heapq.heappush(open_list, neighbor_node)
    return None


def solve_8puzzle_a_star(start_state, goal_state):
    """使用A*算法求解八数码问题"""
    solution = a_star(start_state, goal_state)

    if solution:
        print("=== A* 搜索结果 ===")
        count = 0
        for step in solution:
            count = count + 1
            print(step)
        print("Steps:", count - 1)
    else:
        print("Fail to find a path!")
    print()


# ==================== 主程序 ====================
if __name__ == "__main__":
    start_state_dfs = [1, 4, 2, 8, 6, 3, 7, 0, 5]
    goal_state_dfs = [1, 2, 3, 8, 0, 4, 7, 6, 5]

    start_state_bfs = [1, 4, 2, 8, 6, 3, 7, 0, 5]
    goal_state_bfs = [1, 2, 3, 8, 0, 4, 7, 6, 5]

    start_state_astar = (1, 4, 2, 8, 6, 3, 7, 0, 5)
    goal_state_astar = (1, 2, 3, 8, 0, 4, 7, 6, 5)

    print("=" * 50)
    print("2.3.1 八数码问题")
    print("=" * 50)
    print()

    solve_8puzzle_dfs(start_state_dfs, goal_state_dfs)
    solve_8puzzle_bfs(start_state_bfs, goal_state_bfs)
    solve_8puzzle_a_star(start_state_astar, goal_state_astar)
