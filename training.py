import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kNN import kNN

# Load MNIST digits dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Scale features — kNN works better with normalized features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# Visualize a few examples
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for ax, img, label in zip(axes.ravel(), digits.images, y):
    ax.set_axis_off()
    ax.imshow(img, cmap=plt.cm.gray_r)
    ax.set_title(str(label))
plt.show()

# Initialize and train custom kNN
clf = kNN(k=3)
clf.fit(X_train, y_train)

# Predict test set
predictions = clf.predict(X_test)
acc = np.sum(predictions == y_test) / len(y_test)

print("Predictions:", predictions[:10])
print("Accuracy:", acc)

# Pick a random point from the feature space
feature_mins = X.min(axis=0)
feature_maxs = X.max(axis=0)
rng = np.random.default_rng()
random_point = rng.uniform(feature_mins, feature_maxs)
pred_label_random = clf.predict_one(random_point)
print("Random point predicted class:", int(pred_label_random))

# Visualize a single test sample and prediction
sample_idx = 0
sample_image = X_test[sample_idx].reshape(8, 8)
sample_label = predictions[sample_idx]

plt.imshow(sample_image, cmap=plt.cm.gray_r)
plt.title(f"Predicted: {sample_label}, True: {y_test[sample_idx]}")
plt.axis('off')
plt.show()
