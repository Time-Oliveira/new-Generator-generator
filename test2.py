import numpy as np

# 生成一个3x3的随机整数矩阵
random_matrix = np.mat(np.random.randint(0, 10, size=(3, 3)))

# 生成一个3x3的随机浮点数矩阵（0-1之间）
random_float_matrix = np.mat(np.random.rand(3, 3))

# 输出矩阵
print(random_matrix)
print(random_float_matrix)