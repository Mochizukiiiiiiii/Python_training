#Pythonではじめる機械学習

#%%
import numpy as np
from scipy import sparse

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
