#Pythonではじめる機械学習

#%%
import sys
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
import IPython
import sklearn
from sklearn import neighbors

# %%
eye = np.eye(4)
print("Numpy array:\n{}".format(eye))

#%%
sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))
#疲れたから基本部分は後回しか無視

#%%
from sklearn.datasets import load_iris

iris_dataset = load_iris()

# %%
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# %%
print(iris_dataset['DESCR'][:193] +  "\n...")

# %%
print("Target names: {}".format(iris_dataset['target_names']))

#%%
print("Feature names: {}".format(iris_dataset['feature_names']))

# %%
print("Type of data: {}".format(type(iris_dataset['data'])))

# %%
print("Shape of data: {}".format(iris_dataset['data'].shape))

# %%
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))

# %%
print("Type of target: {}".format(type(iris_dataset['target'])))

# %%
print("Shape of target: {}".format(iris_dataset['target'].shape))

# %%
print("Target:\n{}".format(iris_dataset['target']))

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# %%
print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))

# %%
print('X_test shape: {}'.format(X_test.shape))
print('y_test shape: {}'.format(y_test.shape))
# %%
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# %%
#なんかうまくプロットできない
import pandas as pd
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins:20'}, s=60, alpha=0.8, cmap=mglearn.cm3)
# %%
#k-最近傍法
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# %%
knn.fit(X_train, y_train)
# %%
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=1, n_neighbors=1,p=2, weights='uniform')
# %%
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
# %%
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]
))
# %%
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

# %%
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
# %%
#訓練と評価を行うための最小の手順
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# %%
#2章 教師あり学習
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
# %%
X, y = mglearn.datasets.make_wave(n_samples = 40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")
# %%
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
# %%
print("Shape of cancer data: {}".format(cancer.data.shape))
# %%
print("Sample counts per class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
))
# %%
print("Feature names:\n{}".format(cancer.feature_names))
# %%
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: \n{}".format(boston.data.shape))
# %%
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))
# %%
mglearn.plots.plot_knn_classification(n_neighbors=1)
# %%
mglearn.plots.plot_knn_classification(n_neighbors=3)
# %%
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# %%
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
# %%
clf.fit(X_train, y_train)
# %%
print("Test set predictions: {}".format(clf.predict(X_test)))
# %%
print("Test set acuracy: {:.2f}".format(clf.score(X_test, y_test)))
# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
# %%
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    #モデル構築
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    #訓練セット精度を記録
    training_accuracy.append(clf.score(X_train, y_train))
    #汎化精度を記録
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
# %%
mglearn.plots.plot_knn_regression(n_neighbors=1)
# %%
mglearn.plots.plot_knn_regression(n_neighbors=3)
# %%
from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples = 40)
#waveデータセットを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#3つの最近傍点を考慮するように設定してモデルのインスタンスを生成
reg = KNeighborsRegressor(n_neighbors=3)
#訓練データと訓練ターゲットを用いてモデルを学習させる
reg.fit(X_train, y_train)
# %%
print("Test set predictions:\n{}".format(reg.predict(X_test)))
# %%
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))
# %%
fig, axes = plt.subplots(1, 3, figsize = (15, 4))
#-3から3までの間に1000点のデータポイントを作る
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 1, 3, 9近傍点で予測
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize = 8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(0), markersize = 8)

    ax.set_title(
        "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
            n_neighbors, reg.score(X_train, y_train),
            reg.score(X_test, y_test)))
    ax.set_label("Feature")
    ax.set_label("Target")
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
# %%
mglearn.plots.plot_linear_regression_wave()
# %%
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples = 60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
# %%
print("lr_coef_: {}".format(lr.coef_))
print("lr_intercept_: {}".format(lr.intercept_))
# %%
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
# %%
X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
# %%
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
# %%
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
# %%
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))
# %%
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
# %%
plt.plot(ridge.coef_, 's', label = "Ridge alpha = 1")
plt.plot(ridge10.coef_, '^', label = "Ridge alpha = 10")
plt.plot(ridge01.coef_, 'v', label = "Ridge alpha = 0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magunitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
# %%
mglearn.plots.plot_ridge_n_samples()
# %%
