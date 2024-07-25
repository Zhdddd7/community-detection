import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

# 生成50个样本，每个样本有20个特征
data = np.random.rand(50, 20)

# 生成50个样本的标签，标签为1到5的整数
labels = np.random.randint(1, 6, 50)
target_dir = "./models"
# 创建随机森林分类器
clf = RandomForestClassifier(n_estimators=100)
clf.fit(data, labels)


# 如果你想导出随机森林中的所有树，可以循环遍历所有树并保存每棵树的dot文件
for i, tree in enumerate(clf.estimators_):
    if i >= 5:
        break
    dot_data = export_graphviz(tree, out_file=None,
                               feature_names=[f'feature_{i}' for i in range(data.shape[1])],
                               class_names=[str(i) for i in range(1, 6)],
                               filled=True, rounded=True,
                               special_characters=True)
    with open(f"{target_dir}/random_forest_tree_{i}.dot", "w") as f:
        f.write(dot_data)
