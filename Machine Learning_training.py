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
