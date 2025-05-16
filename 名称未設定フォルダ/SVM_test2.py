import scipy.stats
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import svm,metrics
import pandas as pd

df = pd.read_csv('winequality-white-re.csv')

#条件設定
max_score = 0
SearchMethod = 0

x = df.iloc[:,0:10]
y = df.iloc[:,11]

#PCA
#x=pd.DataFrame(df[["residual sugar","free sulfur dioxide","total sulfur dioxide","alcohol"]])
#UMAP
#x=pd.DataFrame(df[["fixed acidity","volatile acidity","chlorides","total sulfur dioxide","density","alcohol"]])

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size= 0.9,random_state=0,stratify=y)

params =[
    {"C": [1,10,100,1000],"kernel": ["linear"]},
    {"C": [1,10,100,1000],"kernel": ["rbf"],"gamma":[0.001,0.0001]},
    {"C": [1,10,100,1000],"kernel": ["poly"],"gamma":[0.001,0.0001]},
    {"C": [1,10,100,1000],"kernel": ["sigmoid"],"gamma":[0.001,0.0001]},
]

# グリッドサーチの設定と実行
clf = GridSearchCV(svm.SVC(), params, cv=3)
clf.fit(x_train, y_train)

# 最適モデルの表示
print("学習モデル = ", clf.best_estimator_)

# 予測と評価
pre = clf.predict(x_test)
ac_score = metrics.accuracy_score(y_test, pre)
print("正解率 = ", ac_score)