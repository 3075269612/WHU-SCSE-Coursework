# 3.3.1 命题逻辑案例 - 三老师授课分配问题
from itertools import permutations

# 定义老师和课程
teachers = ['A', 'B', 'C']
subjects = ['Chinese', 'Math', 'Politics', 'Geography', 'Music', 'Art']

def check_conditions(assignment):
    """
    检查特定的课程分配是否满足所有逻辑条件。
    :param assignment: 字典，键为老师('A', 'B', 'C')，值为分配的两门课程元组。
    """
    # 政治老师和数学老师是邻居，说明政治和数学不可能是同一个老师教
    for t in teachers:
        if 'Politics' in assignment[t] and 'Math' in assignment[t]:
            return False

    # 地理老师比语文老师年龄大，说明地理和语文不可能是同一个老师教
    for t in teachers:
        if 'Geography' in assignment[t] and 'Chinese' in assignment[t]:
            return False

    # B 最年轻，且地理老师比语文老师年龄大，说明 B 不是地理老师
    if 'Geography' in assignment['B']:
        return False

    # A 经常给地理老师和数学老师讲他看过的文学作品，说明 A 不可能是地理老师和数学老师
    if 'Geography' in assignment['A'] or 'Math' in assignment['A']:
        return False

    # B 经常和音乐老师、语文老师一起游泳，说明 B 不可能是音乐老师和语文老师
    if 'Music' in assignment['B'] or 'Chinese' in assignment['B']:
        return False

    return True

def main():
    """
    主函数：尝试所有可能的课程分配并找到满足条件的解。
    """
    # permutations(subjects) 会生成 6 门课的所有排列
    for perm in permutations(subjects):
        # A, B, C 分别教两门课
        assignments = {
            'A': perm[0:2], 
            'B': perm[2:4], 
            'C': perm[4:6]
        }
        if check_conditions(assignments):
            print("A teaches:", assignments['A'])
            print("B teaches:", assignments['B'])
            print("C teaches:", assignments['C'])
            break

if __name__ == '__main__':
    main()