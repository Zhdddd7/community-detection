import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# 生成样本数据
np.random.seed(0)
data = np.random.rand(50, 20)
labels = np.random.randint(1, 6, 50)  # 假设有3个类别

# 假设我们有3个聚类，每个聚类包含一组特征
clusters = {
    'cluster_1': [0, 1, 2, 3, 4],
    'cluster_2': [5, 6, 7, 8, 9],
    'cluster_3': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
}

# 创建新的特征作为每个聚类的平均值
cluster_features = np.zeros((data.shape[0], len(clusters)))
for i, (cluster_name, feature_indices) in enumerate(clusters.items()):
    cluster_features[:, i] = data[:, feature_indices].mean(axis=1)

data_with_clusters = np.column_stack((data, cluster_features))

# 创建决策树分类器
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(cluster_features, labels)

# 导出决策树
feature_names = [f'cluster_{i+1}' for i in range(len(clusters))]
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=feature_names,
                           class_names=[str(i) for i in range(1, 6)],
                           filled=True, rounded=True,
                           special_characters=True)

# 可视化决策树

target_dir = "./models"
with open(f"{target_dir}/multiDecisionTree.dot", "w") as f:
        f.write(dot_data)