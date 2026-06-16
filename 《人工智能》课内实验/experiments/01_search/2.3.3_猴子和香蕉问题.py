"""
2.3.3 猴子和香蕉问题 (Monkey and Banana)
==========================================
使用 BFS 求解经典的"猴子和香蕉"AI规划问题。

问题描述:
  一个 3×3 的房间中有一只猴子和一个箱子，香蕉挂在天花板 (1,1) 位置。
  猴子够不到香蕉，必须先站在箱子上才能拿到。
  猴子可以:
    1. 在房间内自由走动 (move)
    2. 推箱子到相邻格子 (push box)，前提是猴子必须站在箱子旁并向箱子方向推
    3. 爬上箱子 (climb box)，前提是猴子与箱子在同一位置

状态表示 ((mx, my), (bx, by), has_banana):
  (mx, my):    猴子当前位置
  (bx, by):    箱子当前位置
  has_banana:  是否已经拿到香蕉 (True/False)

初始状态: ((0, 0), (2, 1), False)  猴子在左上角, 箱子在(2,1)
目标状态: ((1, 1), (1, 1), True)   猴子和箱子都在(1,1), 拿到香蕉
"""

from collections import deque

# 9种可能的动作:
#   - 4种普通移动 (猴子独自走动)
#   - 4种推箱子   (猴子+箱子同步移动, 前提 monkey==box 且推的方向合法)
#   - 1种爬箱子   (拿到香蕉, 前提 monkey==box)
MOVES = ['move left', 'move right', 'move up', 'move down',
         'push box left', 'push box right', 'push box up', 'push box down',
         'climb box']


def is_valid(state, room_size):
    """验证猴子和箱子的坐标是否在房间范围内"""
    (mx, my), (bx, by), _ = state
    return (0 <= mx < room_size and 0 <= my < room_size and
            0 <= bx < room_size and 0 <= by < room_size)


def apply_move(state, move, room_size):
    """执行一个动作，返回新的状态。

    动作分三类:
      - move *:   只移动猴子，箱子不动
      - push box *: 猴子和箱子一起向同一方向移动一格（猴子必须在箱子位置）
      - climb box: 猴子爬上箱子拿到香蕉（猴子必须在箱子位置且在香蕉下方(1,1)）

    room_size 用于边界检查，防止越界。
    """
    monkey, box, has_banana = state
    new_monkey = list(monkey)
    new_box = list(box)

    if move == 'move left':
        new_monkey[1] -= 1
    elif move == 'move right':
        new_monkey[1] += 1
    elif move == 'move up':
        new_monkey[0] -= 1
    elif move == 'move down':
        new_monkey[0] += 1
    elif move == 'push box left' and monkey == box and box[1] > 0:
        new_monkey[1] -= 1
        new_box[1] -= 1
    elif move == 'push box right' and monkey == box and box[1] < room_size - 1:
        new_monkey[1] += 1
        new_box[1] += 1
    elif move == 'push box up' and monkey == box and box[0] > 0:
        new_monkey[0] -= 1
        new_box[0] -= 1
    elif move == 'push box down' and monkey == box and box[0] < room_size - 1:
        new_monkey[0] += 1
        new_box[0] += 1
    elif move == 'climb box' and monkey == box:
        # 爬上箱子即拿到香蕉，直接返回成功状态
        return (tuple(new_monkey), tuple(new_box), True)

    new_state = (tuple(new_monkey), tuple(new_box), has_banana)
    return new_state if is_valid(new_state, room_size) else state


def bfs(start, goal, room_size):
    """BFS搜索。

    每次从队列中取出一个状态，依次尝试9种动作。
    由于所有动作代价相同（均为1步），BFS天然保证找到最优解
    （步数最少的动作序列）。
    """
    queue = deque([(start, [])])  # (当前状态, 到达此状态的动作序列)
    visited = set()

    while queue:
        state, path = queue.popleft()

        if state == goal:
            return path

        visited.add(state)

        for move in MOVES:
            new_state = apply_move(state, move, room_size)
            if new_state not in visited:
                queue.append((new_state, path + [move]))
    return None


if __name__ == "__main__":
    print("=" * 50)
    print("2.3.3 猴子和香蕉问题")
    print("=" * 50)
    print()

    start_state = ((0, 0), (2, 1), False)  # 猴(0,0), 箱(2,1), 未拿到
    goal_state = ((1, 1), (1, 1), True)    # 猴(1,1), 箱(1,1), 拿到
    room_size = 3

    solution = bfs(start_state, goal_state, room_size)

    if solution:
        print("Moves:")
        count = 0
        for step in solution:
            count = count + 1
            print(step)
        print("Steps:", count - 1)  # 减去 climb box (第5步到达目标)
    else:
        print("Fail to find a path!")
