"""
2.3.5 0-1背包问题 (0-1 Knapsack Problem)
==========================================
使用 DFS 穷举搜索求解 0-1 背包问题。

问题描述:
  有 n 件物品，每件物品有重量 w[i] 和价值 v[i]。
  背包容量为 C，每件物品只能选或不选（0或1，不能选半个）。
  目标: 在总重量不超过 C 的前提下，使总价值最大化。

  本例数据:
    物品重量: [1, 3, 2, 5]
    物品价值: [1, 8, 10, 15]
    背包容量: 7

  理论最优解: 选第3件(重量2,价值10) + 第4件(重量5,价值15)
              总重量=7, 总价值=25

算法思路 —— 二叉决策树 DFS:
  每个物品有"选"和"不选"两个分支，形成一棵深度为 n 的满二叉树。
  从根到叶每一条路径代表一种选择方案。
  DFS 遍历整棵二叉树，用全局变量记录遇到的最大价值。
  剪枝条件: 如果当前累计重量已经超过容量，直接返回（不必继续向下）。
"""


def knapsack(weights, values, capacity):
    """使用DFS遍历所有子集组合，返回最大价值。

    时间复杂度: O(2^n) —— 每个物品有选/不选两种可能
    搜索空间: 2^4 = 16 种组合（本题 n=4）
    """
    n = len(weights)
    max_value = 0  # 非局部变量，在 dfs 外层被修改和读取

    def dfs(index, current_weight, current_value):
        """递归探索第 index 件物品的选与不选。

        参数:
            index:          当前正在决策的物品索引(0 ~ n-1)
            current_weight: 已选物品的累计重量
            current_value:  已选物品的累计价值
        """
        nonlocal max_value

        # 剪枝: 超过容量，此分支无效
        if current_weight > capacity:
            return

        # 更新当前找到的最优解
        if current_value > max_value:
            max_value = current_value

        # 递归终止: 所有物品决策完毕
        if index == n:
            return

        # 分支1: 不选第 index 件物品 — 重量和价值都不变
        dfs(index + 1, current_weight, current_value)

        # 分支2: 选第 index 件物品 — 重量和价值都增加
        dfs(index + 1, current_weight + weights[index], current_value + values[index])

    dfs(0, 0, 0)
    return max_value


if __name__ == "__main__":
    print("=" * 50)
    print("2.3.5 0-1背包问题")
    print("=" * 50)
    print()

    weights = [1, 3, 2, 5]
    values = [1, 8, 10, 15]
    capacity = 7

    max_value = knapsack(weights, values, capacity)
    print("物品重量:", weights)
    print("物品价值:", values)
    print("背包容量:", capacity)
    print("最大价值:", max_value)
