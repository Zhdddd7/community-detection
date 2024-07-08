import numpy as np
import random

def generate_correlated_weights(num_groups, group_size, correlation):
    """
    生成具有特定相关性结构的权重矩阵。

    参数：
    num_groups (int): 组的数量。
    group_size (int): 每组的大小。
    correlation (float): 组内的相关性。

    返回：
    weights (np.ndarray): 具有特定相关性结构的权重矩阵。
    """
    weights = []
    for _ in range(num_groups):
        base_weight = np.random.rand(group_size)
        group_weights = [base_weight + np.random.normal(0, 1-correlation, group_size) for _ in range(group_size)]
        weights.extend(group_weights)
    return np.array(weights)

def generate_linear_function(X, weights, func_num):
    """
    生成一个线性函数，从 X 中挑选任意个数变量进行加权和运算。

    参数：
    X (list): 输入变量列表。
    weights (np.ndarray): 权重数组。
    func_num (int): 函数编号。

    返回：
    func_code (str): 生成的函数代码。
    """
    selected_indices = range(len(X))  # 使用所有变量
    selected_vars = [X[i] for i in selected_indices]

    # 创建显式表达式
    terms = [f'{weights[i]:.2f}*x[{selected_indices[i]}]' for i in range(len(weights))]
    expression = ' + '.join(terms)
    
    func_code = f"def func{func_num}(x):\n    return {expression}"
    
    return func_code

# 生成具有特定相关性结构的权重矩阵
num_groups = 3
group_sizes = [7, 7, 6]
correlation = 0.8
weights = generate_correlated_weights(num_groups, max(group_sizes), correlation)

# 生成 20 个线性函数
X = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
fun_list = []

func_num = 0
for group_id, group_size in enumerate(group_sizes):
    for i in range(group_size):
        func_code = generate_linear_function(X, weights[func_num], func_num)
        fun_list.append(func_code)
        func_num += 1

# 打印生成的函数代码
for func_code in fun_list:
    print(func_code)
    print()

# 示例使用（将生成的函数代码转换为实际函数）
local_vars = {}
exec('\n'.join(fun_list), globals(), local_vars)

x_values = np.random.rand(10) * 10  # 生成随机的 x 值

for i in range(20):
    func = local_vars[f'func{i}']
    result = func(x_values)
    print(f"Function {i}: {func.__name__}(x) = {result}")

# 打印显式表达式
print("\n显式表达式:")
for func_code in fun_list:
    print(func_code)
