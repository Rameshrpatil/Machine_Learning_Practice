import sys

import mglearn as mglearn
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn

# print versions of all essential libraries
#print("Python version: {}".format(sys.version))
#print("Pandas version: {}".format(pd.__version__))
#print("matplotlib version: {}".format(matplotlib.__version__))
#print("NumPy version: {}".format(np.__version__))
#print("SciPy version: {}".format(sp.__version__))
#print("IPython version: {}".format(IPython.__version__))
#print("Scikit-Learn version: {}".format(sklearn.__version__))


from sklearn.datasets import load_iris
iris_dataset = load_iris()

# prints attributes of iris
#print("keys of iris_dataset: \n{} ".format(iris_dataset.keys()))

#print(iris_dataset['DESCR'][:193]+ "\n...")

#print("Target names: {}".format(iris_dataset['target_names']))

#print("Feature names: \n{}".format(iris_dataset['feature_names']))

#print("Type of data: {}".format(type(iris_dataset['data'])))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
 marker='o', hist_kwds={'bins': 20}, s=60,
 alpha=.8, cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
 iris_dataset['target_names'][prediction]))