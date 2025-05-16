import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split

# データの読み込み
df = pd.read_csv('winequality-white-re.csv')

# 条件設定
max_score = 0
SearchMethod = 0

x = df.iloc[:, 0:10]
y = df.iloc[:, 11]

#PCA
#x = pd.DataFrame(df[["residual sugar", "free sulfur dioxide", "total sulfur dioxide", "alcohol"]])
#UMP
x=pd.DataFrame(df[["fixed acidity","volatile acidity","chlorides","total sulfur dioxide","density","alcohol"]])

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=0, stratify=y)

params = [
    {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
    {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]},
    {"C": [1, 10, 100, 1000], "kernel": ["poly"], "gamma": [0.001, 0.0001]},
    {"C": [1, 10, 100, 1000], "kernel": ["sigmoid"], "gamma": [0.001, 0.0001]},
]

'''
# グリッドサーチの設定と実行
clf = GridSearchCV(svm.SVC(), params, cv=3)
clf.fit(x_train, y_train)

# 最適モデルの表示
print("学習モデル = ", clf.best_estimator_)

# 予測と評価
pre = clf.predict(x_test)
ac_score = metrics.accuracy_score(y_test, pre)
print("正解率 = ", ac_score)
'''

# 検証曲線の描画
param_range = [1, 10, 100, 1000]
train_scores, test_scores = validation_curve(
    svm.SVC(kernel='poly'), x, y, param_name="C", param_range=param_range, cv=3, scoring="accuracy", n_jobs=-1
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM (poly kernel)")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
plt.xscale("log")
plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=2)
plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color="darkorange", lw=2)
plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=2)
plt.fill_between(param_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color="navy", lw=2)
plt.legend(loc="best")
plt.show()
