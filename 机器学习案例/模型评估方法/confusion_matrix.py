import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

warnings.filterwarnings('ignore')
np.random.seed(42)

# 保存图片的地址
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(
        thresholds,
        precisions[:-1],
        "b--",
        label="Precision"
    )
    plt.plot(
        thresholds,
        recalls[:-1],
        "g-",
        label="Recall"
    )
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(
        recalls,
        precisions,
        "b-",
        linewidth=2
    )
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)


def get_data():
    """
    Get MNIST data ready to learn with.
    :return:
    """
    # 在sklearn的0.2版本中，fetch_mldata函数已经被fetch_openml函数取代
    from sklearn.datasets import fetch_openml     # 通过名称或数据集ID从openml获取数据集

    # 查询到我电脑上的scikit data home目录
    from sklearn.datasets.base import get_data_home
    print(get_data_home())             # C:\Users\hqs\scikit_learn_data

    # Mnist 数据是图像数据：(28,28,1)的灰度图
    """注意：
    fetch_openml返回的是未排序的MNIST数据集。
    fetch_mldata返回按目标排序的数据集。
    在SciKit-Learn 0.20后已经弃用fetch_mldata(),需要使用fetch_openml()。
    如果要得到和之前相同的结果，需要排序数据集。
    """
    mnist = fetch_openml('mnist_784', version=1, cache=True)   # fetch_openml返回一个未排序的数据集
    mnist.target = mnist.target.astype(np.int8)
    sort_by_target(mnist)
    # print(mnist.data.shape)    # (70000, 784)

    X, y = mnist["data"], mnist["target"]
    print(X.shape)    # (70000, 784)
    print(y.shape)    # (70000,)

    # 切分为训练集和测试集
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    # 洗牌操作，打乱当前数据集顺序
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]   # 索引值回传相当于洗牌操作
    print(X_train, y_train)

    # 训练二分类器
    y_train_5 = (y_train == 5)   # 修改便签为是否等于5
    y_test_5 = (y_test == 5)

    from sklearn.linear_model import SGDClassifier     # 引入线性分类器

    # 使用scikit-learn的SGDClassifier类来创建分类器，区分图片是否是数字5
    sgd_clf = SGDClassifier(
        max_iter=5,       # 训练迭代次数
        tol=-np.infty,
        random_state=42   # 传入随机种子，每次随机结果一样
    )
    # fit方法:用随机梯度下降法拟合线性模型
    sgd_clf.fit(X_train, y_train)

    from sklearn.model_selection import cross_val_predict

    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    print(y_train_pred.shape)     # (60000,)：60000个样本的预测结果
    print(X_train.shape)       # (60000, 784)：训练样本也是60000个，与预测结果数量一致

    from sklearn.metrics import confusion_matrix

    confusion_matrix(y_train_5, y_train_pred)
    """
    array([[53417  1162],
           [1350  4071]], dtype=int64)
    """

    # 准确率（Precision）和召回率（Recall）
    from sklearn.metrics import precision_score, recall_score

    print(precision_score(y_train_5, y_train_pred))   # 0.7779476399770686
    print(recall_score(y_train_5, y_train_pred))      # 0.7509684560044272

    # F1 score
    from sklearn.metrics import f1_score

    print(f1_score(y_train_5, y_train_pred))          # 0.7642200112633752

    # 阈值
    # y_scores = sgd_clf.decision_function([X[35000]])
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    print(y_scores)

    t = 5000
    y_pred = (y_scores > t)
    print(y_pred)

    print(y_train_5.shape)
    print(y_scores.shape)

    from sklearn.metrics import precision_recall_curve

    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    # print(precisions, recalls, thresholds)

    plt.figure(figsize=(8, 4))
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.xlim([-700000, 700000])
    plt.show()

    # 随着recall变化precision的变化情况
    plt.figure(figsize=(8, 6))
    plot_precision_vs_recall(precisions, recalls)
    plt.show()

    # ROC 曲线
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

    plt.figure(figsize=(8, 6))
    plot_roc_curve(fpr, tpr)
    plt.show()

    # AUC曲线下面积
    from sklearn.metrics import roc_auc_score

    print(roc_auc_score(y_train_5, y_scores))      # 0.9562435587387078


get_data()

