import numpy as np


# 定义函数
def func0(x): return 0.40*x[0] + 1.13*x[1] + 0.28*x[2] + 0.36*x[3] + 0.59*x[4] 
def func1(x): return 0.51*x[0] + 1.15*x[1] + 0.52*x[2] + 0.30*x[3] + 0.50*x[4] 
def func2(x): return 0.68*x[0] + 0.88*x[1] + 0.24*x[2] + 0.33*x[3] + 0.05*x[4] 
def func3(x): return 0.53*x[0] + 1.18*x[1] + 0.39*x[2] + 0.79*x[3] + 0.27*x[4] 
def func4(x): return 0.23*x[0] + 1.22*x[1] + 0.29*x[2] + 0.57*x[3] + 0.35*x[4] 
def func5(x): return 0.33*x[0] + 0.61*x[1] + 0.14*x[2] + 0.07*x[3] + 0.63*x[4] + 0.47*x[5] + 0.84*x[6]
def func6(x): return 0.33*x[0] + 1.24*x[1] + 0.30*x[2] + 0.64*x[3] + 0.75*x[4] + 0.23*x[5] + 0.98*x[6]
def func7(x): return 0.27*x[0] + 0.62*x[1] + 0.61*x[2] + -0.07*x[3] + 0.17*x[4] + 0.57*x[5] + 0.41*x[6]
def func8(x): return 0.46*x[0] + 0.83*x[1] + 0.49*x[2] + -0.25*x[3] + 0.31*x[4] + 1.18*x[5] + 0.64*x[6]
def func9(x): return 0.42*x[0] + 0.54*x[1] + 0.17*x[2] + 0.09*x[3] + 0.15*x[4] + 0.76*x[5] + 0.88*x[6]
def func10(x): return 0.59*x[0] + 0.78*x[1] + 0.18*x[2] + 0.11*x[3] + 0.25*x[4] + 0.58*x[5] + 0.68*x[6]
def func11(x): return 0.67*x[0] + 0.86*x[1] + 0.17*x[2] + 0.10*x[3] + 0.44*x[4] + 0.88*x[5] + 0.44*x[6]
def func12(x): return 0.43*x[0] + 0.65*x[1] + 0.12*x[2] + 0.54*x[3] + 0.09*x[4] + 1.03*x[5] + 0.64*x[6]
def func13(x): return 0.49*x[0] + 0.99*x[1] + 0.53*x[2] + -0.08*x[3] + 0.32*x[4] + 0.45*x[5] + 0.62*x[6]
def func14(x): return 0.57*x[0] + 0.99*x[1] + -0.04*x[2] + 0.28*x[3] + 0.57*x[4] + 0.60*x[5] + 0.45*x[6]
def func15(x): return 0.89*x[0] + 0.63*x[1] + -0.14*x[2] + 1.08*x[3] + 1.16*x[4] + 1.09*x[5] + 0.73*x[6]
def func16(x): return 0.46*x[0] + 0.62*x[1] + 0.01*x[2] + 0.66*x[3] + 0.91*x[4] + 0.63*x[5] + 0.29*x[6]
def func17(x): return 0.69*x[0] + 1.08*x[1] + 0.04*x[2] + 0.54*x[3] + 0.40*x[4] + 0.70*x[5] + 0.98*x[6]
def func18(x): return 0.80*x[0] + 0.31*x[1] + -0.02*x[2] + 0.46*x[3] + 0.78*x[4] + 0.83*x[5] + -0.06*x[6]
def func19(x): return 0.74*x[0] + 0.74*x[1] + -0.00*x[2] + 0.32*x[3] + 0.98*x[4] + 0.86*x[5] + 0.49*x[6]

functions = [func1, func2, func3, func4, func5, func6, func7, func8, func9, func10,
             func11, func12, func13, func14, func15, func16, func17, func18, func19, func0]


import numpy as np
sample_dim = 100
epsilon = 0.05
x = np.array([1] * sample_dim)

# [x1, x2,...xm]
# 每一个x服从正态分布
m = 10
n = 20 # 特征数
samples = np.zeros((m, sample_dim))
print("the sample dimension:", samples.shape)
for i in range(samples.shape[0]):
    samples[i] = np.random.randn(sample_dim)
res = []
for func in functions:
    res.append(func(samples))
res = np.array(res)

from communityDetection import Community
labels = [f'feature_{i}' for i in range(n)]
res = res.T
print("the res dimension:", res.shape)
test = Community(res, labels)
cov_matrix = test.get_cor_matrix(True)
from modified_louvain import *
com_list = Louvain_cov(cov_matrix)
# com_list = test.partition(cov_matrix)
print("----Modified-Louvain的社区检测结果为----")
print(com_list)
print("----Networkx-Louvain的社区检测结果为----")
print(test.partition(cov_matrix))
# res = test.get_cor_matrix(True)
communities = test.BGLL(epsilon)
# communities_2 = test.BGLL(epsilon, target_communities)
print("----DIY-Louvain检测结果为----")
print(communities)