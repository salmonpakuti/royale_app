import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score

class SVCModel:
    """
    このクラスは、SVCでワインの特徴量を学習、グリッドサーチでパラメータ調整、検証曲線の表示を行うクラスです。
    """
    
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        
        # PCAを使用する場合
        # self.x = self.df[["residual sugar", "free sulfur dioxide", "total sulfur dioxide", "alcohol"]]
        
        # UMAPを使用する場合
        # self.x = self.df[["fixed acidity", "volatile acidity", "chlorides", "total sulfur dioxide", "density", "alcohol"]]
        
        # そのままの特徴量を使用する場合
        self.x = self.df.iloc[:, 0:10]
        self.y = self.df.iloc[:, 11]
    
    def svc_learning(self):
        """
        SVCでワインの品質を学習を行い、精度を出力するメソッド。
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, shuffle=True, random_state=3, stratify=self.y)
        
        model = svm.SVC(C=100, kernel='poly', gamma=0.001)
        model.fit(x_train, y_train)
        
        pred_train = model.predict(x_train)
        accuracy_train = accuracy_score(y_train, pred_train)
        print('トレーニングデータに対する正解率: %.2f' % accuracy_train)
        
        pred_test = model.predict(x_test)
        accuracy_test = accuracy_score(y_test, pred_test)
        print('テストデータに対する正解率: %.2f' % accuracy_test)
    
    def svc_gridsearch(self):
        """
        SVCのグリッドサーチを行い、最適なパラメータを出力するメソッド。
        """
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.3, shuffle=True, random_state=3, stratify=self.y)
        
        params = [
            {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
            {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]},
            {"C": [1, 10, 100, 1000], "kernel": ["poly"], "gamma": [0.001, 0.0001]},
            {"C": [1, 10, 100, 1000], "kernel": ["sigmoid"], "gamma": [0.001, 0.0001]},
        ]
        
        clf = GridSearchCV(svm.SVC(), params, cv=3)
        clf.fit(x_train, y_train)
        
        print("最適な学習モデル: ", clf.best_estimator_)
        
        pre = clf.predict(x_test)
        ac_score = metrics.accuracy_score(y_test, pre)
        print("テストデータに対する正解率: ", ac_score)
    
    def plot_validation_curve(self):
        """
        検証曲線を描画するメソッド。
        """
        param_range = [1, 10, 100, 1000]
        train_scores, test_scores = validation_curve(
            svm.SVC(kernel='poly'), self.x, self.y, param_name="C", param_range=param_range, cv=3, scoring="accuracy", n_jobs=-1
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

if __name__ == "__main__":
    filepath = "/Users/adachiharuto/prog/名称未設定フォルダ/winequality-white-re.csv"
    model = SVCModel(filepath)
    model.svc_learning()
    model.svc_gridsearch()
    model.plot_validation_curve()
