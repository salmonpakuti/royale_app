from sklearn import datasets #予め用意されてるデータセット読み込み用ライブラリ
iris = datasets.load_iris() #datasets.load[tab]
data = iris.data
target = iris.target
print("#######################################")
print(iris.target_names)