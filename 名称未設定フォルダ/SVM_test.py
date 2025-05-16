from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import svm,metrics
import scipy.stats
from sklearn.datasets import load_breast_cancer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

class svc:
    """_summary_
        このクラスは、svcでワインの特徴量を学習、グリッドサーチでパラメータ調整、検証曲線の表示を行うクラスです。
        filepath(str):データセットのファイルパス
        df(dataframe):csvファイルを読み込み、データフレーム型として格納します
        x(dataframe):特徴量
        y(dataframe):正解ラベル
        model:学習で利用するモデル
    """



    df = pd.read_csv('winequality-white-re.csv')
    
    #そのまま
    x = df.iloc[:,0:10]
    #PCA
    x=pd.DataFrame(df[["residual sugar","free sulfur dioxide","total sulfur dioxide","alcohol"]])
    #UMAP
    x=pd.DataFrame(df[["fixed acidity","volatile acidity","chlorides","total sulfur dioxide","density","alcohol"]])

    y = df.iloc[:,11]

    def svc_learning(a,b):
                """_summary_：svcでワインの品質を学習を行い、精度を出力するメソッド
                Args:
                    a (dataframe): 特徴量
                    b (dataframe): 正解ラベル
                """
                #XとYを学習データとテストデータに分割
                x_train,x_test,y_train,y_test = train_test_split(a,b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
                y_train=np.reshape(y_train,(-1))
                y_test=np.reshape(y_test,(-1))

                model = svm.SVC(C=100,kernel='poly',gamma=0.001)
                model.fit(x_train,y_train)
                pred_train = model.predict(x_train)
                accuracy_train = accuracy_score(y_train,pred_train)
                print('トレーニングデータに対する正解率:%.2f'% accuracy_train)
                pred_test = model.predict(x_test)
                accuracy_test = accuracy_score(y_test , pred_test)
                print('テストデータに対する正解率:%.2f'% accuracy_test)


    def svc_gridsearch(a,b):
        """_summary_：svcのグリッドサーチを行い、最適なパラメータを出力するメソッド
        Args:
        a (dataframe): 特徴量
        b (dataframe): 正解ラベル
        """

        max_score = 0
        SearchMethod = 0

        #XとYを学習データとテストデータに分割
        x_train,x_test,y_train,y_test = train_test_split(a,b, test_size=0.3, shuffle=True, random_state=3, stratify=b)
        y_train=np.reshape(y_train,(-1))
        y_test=np.reshape(y_test,(-1))
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

    def plot_validation_curve(a,b):
        """_summary_：検証曲線を描画するメソッド
            Args:
            a (dataframe): 特徴量
            b (dataframe): 正解ラベル
        """
         # 検証曲線の描画
        param_range = [1, 10, 100, 1000]
        train_scores, test_scores = validation_curve(
        svm.SVC(kernel='poly'), a, b, param_name="C", param_range=param_range, cv=3, scoring="accuracy", n_jobs=-1
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
