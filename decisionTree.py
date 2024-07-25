import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# 生成50个样本，每个样本有20个特征
data = np.random.rand(50, 20)

# 生成50个样本的标签，标签为1到5的整数
labels = np.random.randint(1, 6, 50)

clf = DecisionTreeClassifier()
clf.fit(data, labels)

# 导出决策树
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=[f'feature_{i}' for i in range(data.shape[1])],
                           class_names=[str(i) for i in range(1, 6)],
                           filled=True, rounded=True,
                           special_characters=True)

# 可视化决策树
target_dir = "./models"
with open(f"{target_dir}/decisionTree.dot", "w") as f:
        f.write(dot_data)


