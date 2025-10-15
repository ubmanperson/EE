import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from kNN import kNN

# Fix color code for blue (missing #)
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

plt.figure()
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolor='k', s=20)

clf = kNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)

# Generate a random 4D point within the observed feature ranges
feature_mins = X.min(axis=0)
feature_maxs = X.max(axis=0)
rng = np.random.default_rng()
random_point = rng.uniform(feature_mins, feature_maxs)
pred_label_random = clf.predict_one(random_point)
print("Random point:", random_point, "Predicted class:", int(pred_label_random))

# Overlay the random point using its petal length/width (features 2 and 3)
plt.scatter([random_point[2]], [random_point[3]], c=[pred_label_random], cmap=cmap,
            edgecolors='k', linewidths=1.5, s=200)
plt.title("Iris dataset with predicted random point")
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.show()