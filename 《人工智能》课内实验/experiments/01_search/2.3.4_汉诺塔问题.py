"""
2.3.4 汉诺塔问题 (Tower of Hanoi)
===================================
使用递归（本质是DFS）求解经典汉诺塔问题。

问题描述:
  有三根柱子 A、B、C，A柱上从下到上套着 n 个大小不同的圆盘
  （大的在下面，小的在上面）。目标是将所有盘子从 A 柱移到 C 柱。
  规则:
    1. 每次只能移动一个盘子（最上面的那个）
    2. 任何时刻大盘不能放在小盘上面
    3. B 柱可以作为辅助柱

递归思路 (分治法):
  对于 n 个盘子从 source 移到 target (auxiliary 为辅助):
    第1步: 将上面 n-1 个小盘子从 source 移到 auxiliary (用 target 辅助)
    第2步: 将最底下的大盘子直接从 source 移到 target
    第3步: 将 n-1 个小盘子从 auxiliary 移到 target (用 source 辅助)

  递推公式: T(n) = 2T(n-1) + 1
  显式解:   T(n) = 2^n - 1
  如 n=3 时最少需要 2³-1 = 7 步

  这也是对状态空间的深度优先搜索:
    每个状态有若干合法移动，递归选择一条路走到黑(移动所有盘子)，
    然后回溯处理其他选择。
"""


def dfs_hanoi(n, source, target, auxiliary, moves):
    """递归求解 n 层汉诺塔。

    参数:
        n:         要移动的盘子数
        source:    源柱子标识 (如 'A')
        target:    目标柱子标识 (如 'C')
        auxiliary: 辅助柱子标识 (如 'B')
        moves:     记录移动步骤的列表（就地追加）

    递归过程:
        n=0: 空操作，递归终止条件
        n>0:
          (a) 把 n-1 个盘子从 source 通过 target 搬到 auxiliary
          (b) 把第 n 号盘子(最大的)直接从 source 搬到 target
          (c) 把 n-1 个盘子从 auxiliary 通过 source 搬到 target
    """
    if n == 0:
        return
    # Step 1: 顶部 n-1 个盘子: source → auxiliary (target 辅助)
    dfs_hanoi(n - 1, source, auxiliary, target, moves)
    # Step 2: 底部第 n 号盘子: source → target
    moves.append(f"Move disk {n} from {source} to {target}")
    # Step 3: n-1 个盘子: auxiliary → target (source 辅助)
    dfs_hanoi(n - 1, auxiliary, target, source, moves)


def solve_hanoi_dfs(n):
    """入口函数，返回 n 盘汉诺塔的最优移动步骤列表。"""
    moves = []
    dfs_hanoi(n, 'A', 'C', 'B', moves)
    return moves


if __name__ == "__main__":
    print("=" * 50)
    print("2.3.4 汉诺塔问题")
    print("=" * 50)
    print()

    n = 3
    solution = solve_hanoi_dfs(n)
    for step in solution:
        print(step)
    print("Total moves:", len(solution))
