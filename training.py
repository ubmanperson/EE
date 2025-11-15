import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kNN import kNN

# Optional: EMNIST (letters)
import torch
from torchvision import datasets as tv_datasets, transforms

# -------------------------------
# CONFIG — Choose dataset here
# -------------------------------
USE_EMNIST_LETTERS = True   # False → MNIST digits; True → EMNIST letters

# -------------------------------
# Load dataset
# -------------------------------
if USE_EMNIST_LETTERS:
    print("Loading EMNIST Letters...")
    transform = transforms.Compose([
        transforms.ToTensor(),              # Convert to tensor
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 to 784
    ])

    train_set = tv_datasets.EMNIST(
        root='./data', split='letters', train=True,
        download=True, transform=transform
    )
    test_set = tv_datasets.EMNIST(
        root='./data', split='letters', train=False,
        download=True, transform=transform
    )

    X_train = train_set.data.view(len(train_set), -1).numpy().astype(np.float32)
    y_train = train_set.targets.numpy()
    X_test = test_set.data.view(len(test_set), -1).numpy().astype(np.float32)
    y_test = test_set.targets.numpy()

    # Normalize pixel values [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    print(f"EMNIST letters loaded. Train: {X_train.shape}, Test: {X_test.shape}")

else:
    print("Loading MNIST Digits (scikit-learn)...")
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    print(f"Digits loaded. Train: {X_train.shape}, Test: {X_test.shape}")

# -------------------------------
# Train & evaluate custom kNN model
# -------------------------------

k_values = [1, 2, 3, 4, 5, 6, 7, 8, 10]
# Limit sample sizes so we do not accidentally ask for more items
if USE_EMNIST_LETTERS:
    sample_sizes = [100, 200, 500, 1000, 2000]
else:
    # Scikit-learn digits has only 360 test samples; pick smaller chunks
    sample_sizes = [50, 100, len(X_test)]

for k in k_values:
    print(f"\nTraining kNN with k={k}")
    clf = kNN(k=k)
    clf.fit(X_train, y_train)

    for n in sample_sizes:
        n_samples = min(n, len(X_test))
        idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[idx]
        y_sample = y_test[idx]
        predictions = clf.predict(X_sample)
        acc = np.sum(predictions == y_sample) / len(y_sample)
        print(f"Accuracy for k={k}, n={n_samples}: {acc*100:.2f}%")
