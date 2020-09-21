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

    # predict方法:预测当前的结果
    sgd_clf.predict([X[35000]])

    # 采用准确率为衡量指标查看交叉验证的结果
    from sklearn.model_selection import cross_val_score
    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
    """
    [0.96225 0.9645  0.94765]
    """

    # StratifiedKFold方法：按自己的想法平均切割数据集
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone         # 引入克隆可以在估算器中对模型进行深层复制，构造一个具有相同参数的新估算器

    skfolds = StratifiedKFold(
        n_splits=3,
        random_state=42     # 设置随机种子
    )
    for train_index, test_index in skfolds.split(X_train, y_train_5):     # 切割训练的数据集和标签集
        clone_clf = clone(sgd_clf)    # 克隆构建模型
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_folds = X_train[test_index]
        y_test_folds = y_train_5[test_index]

        # fit方法:用随机梯度下降法拟合线性模型
        clone_clf.fit(X_train_folds, y_train_folds)
        # 预测
        y_pred = clone_clf.predict(X_test_folds)
        # 做对了的个数
        n_correct = sum(y_pred == y_test_folds)
        print(n_correct / len(y_pred))
        """
        0.96225
        0.9645
        0.94765
        """


get_data()
