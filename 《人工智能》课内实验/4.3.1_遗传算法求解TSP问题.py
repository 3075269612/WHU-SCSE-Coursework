# 4.3.1 遗传算法案例 - 解决旅行商问题(TSP)
import random

# 定义城市列表和距离矩阵
city_list = ['A', 'B', 'C', 'D', 'E']
distance_matrix = {
    'A': {'A': 0, 'B': 2, 'C': 9, 'D': 10, 'E': 5},
    'B': {'A': 2, 'B': 0, 'C': 6, 'D': 4, 'E': 8},
    'C': {'A': 9, 'B': 6, 'C': 0, 'D': 3, 'E': 7},
    'D': {'A': 10, 'B': 4, 'C': 3, 'D': 0, 'E': 6},
    'E': {'A': 5, 'B': 8, 'C': 7, 'D': 6, 'E': 0}
}

# 遗传算法参数设置
population_size = 50   # 种群数量
elite_size = 10        # 选择个体数量 (精英数目)
mutation_rate = 0.01   # 突变概率
generations = 100      # 迭代次数

def create_individual(city_list):
    """创建一个随机个体（随机路径）"""
    return random.sample(city_list, len(city_list))

def create_population(city_list, population_size):
    """创建初始种群"""
    population = []
    for _ in range(population_size):
        population.append(create_individual(city_list))
    return population

def calculate_fitness(individual):
    """适应度函数，计算路径的总距离"""
    total_distance = 0
    for i in range(len(individual)-1):
        city1 = individual[i]
        city2 = individual[i+1]
        total_distance += distance_matrix[city1][city2]
    return total_distance

def select_elite(population, elite_size):
    """选择精英个体"""
    # 距离越小适应度越高，从小到大排序
    population_fitness = [(individual, calculate_fitness(individual)) for individual in population]
    population_fitness = sorted(population_fitness, key=lambda x: x[1])
    elite = [individual for individual, _ in population_fitness[:elite_size]]
    return elite

def crossover(parent1, parent2):
    """交叉操作，产生子代"""
    child = [None] * len(parent1)
    geneA = random.randint(0, len(parent1)-1)
    geneB = random.randint(0, len(parent1)-1)
    start_gene = min(geneA, geneB)
    end_gene = max(geneA, geneB)
    
    # 继承父代1的一段基因
    for i in range(start_gene, end_gene+1):
        child[i] = parent1[i]
        
    # 其余部分从父代2中按顺序填入，避免重复城市
    for i in range(len(parent2)):
        if parent2[i] not in child:
            for j in range(len(child)):
                if child[j] is None:
                    child[j] = parent2[i]
                    break
    return child

def mutate(individual):
    """变异操作：以一定概率交换染色体中的两个基因(城市)"""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual)-1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

def evolve_population(population, elite_size):
    """执行一代群体进化操作"""
    elite = select_elite(population, elite_size)
    population_size = len(population)
    children = []
    
    while len(children) < population_size:
        parent1 = random.choice(elite)
        parent2 = random.choice(elite)
        child = crossover(parent1, parent2)
        child = mutate(child)
        children.append(child)
        
    return children

def main():
    population = create_population(city_list, population_size)
    
    for i in range(generations):
        population = evolve_population(population, elite_size)
        
    # 计算并找到最后一代种群内的最佳路径
    best_individual = min(population, key=calculate_fitness)
    best_distance = calculate_fitness(best_individual)
    
    print("最佳路径：", best_individual)
    print("最短距离：", best_distance)

if __name__ == '__main__':
    main()