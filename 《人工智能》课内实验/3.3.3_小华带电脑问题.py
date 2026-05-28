# 3.3.3 归结推理案例 - 小华带笔记本电脑问题
import os

S = []  # 用于存储读取的内容

def read_clause_set(filepath):
    """
    读取子句集文件并将每一行转化为列表存放。
    """
    global S
    for line in open(filepath, mode='r', encoding='utf-8'):
        line = line.replace(' ', '').strip()
        if line:
            line = line.split('v')
            S.append(line)

def opposite(clause):
    """
    对命题变项进行取反操作。
    :param clause: 命题变项字符串
    """
    if '~' in clause:
        return clause.replace('~', '')
    else:
        return '~' + clause

def resolution():
    """
    归结函数：对子句集 S 进行归结演算法证明
    """
    global S
    X = S.copy()
    end = False
    
    while True:
        if end or len(X) == 0:
            break
        father = X.pop()
        
        for i in father[:]:
            if end: break
            
            for mother in X[:]:
                if end: break
                
                # 寻找能和 father 中的子句发生归结的相反项
                j = list(filter(lambda x: x == opposite(i), mother))
                if j == []:
                    continue
                else:
                    print('\n亲本子句：' + ' v '.join(father) + ' 和 ' + ' v '.join(mother))
                    father.remove(i)
                    mother.remove(j[0])
                    
                    if (father == [] and mother == []):
                        print('归结式：NIL')
                        end = True
                    elif father == []:
                        print('归结式：' + ' v '.join(mother))
                    elif mother == []:
                        print('归结式：' + ' v '.join(father))
                    else:
                        print('归结式：' + ' v '.join(father) + ' v ' + ' v '.join(mother))

def main():
    filePath = 'S.txt'
    
    # 为了保证代码独立运行，自动生成 S.txt 提供推理约束文件
    if not os.path.exists(filePath):
        with open(filePath, 'w', encoding='utf-8') as f:
            f.write("~P v Q\n")
            f.write("~U v ~T v P\n")
            f.write("U\n")
            f.write("T\n")
            f.write("~Q\n")

    read_clause_set(filePath)
    resolution()
    
    result = list(filter(None, S))
    if result:
        result_list = [item for l in result for item in l]
        print('\n归结结果:' + ' v '.join(result_list))

if __name__ == '__main__':
    main()