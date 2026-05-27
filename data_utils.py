import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def get_standard_data():
    """Генерирует стандартный датасет и выполняет Z-score стандартизацию."""
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Стандартизация (Z-score)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test, X


def generate_custom_data(n_samples=500, dataset_type='linear', noise_prob=0.05):
    """Генератор кастомных данных: linear, xor, circle."""
    if dataset_type == 'linear':
        X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
        X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
        X = np.vstack([X1, X2])
        y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples // 2)])

    elif dataset_type == 'xor':
        X = np.random.randn(n_samples, 2) * 0.5
        X[:n_samples // 4] += np.array([2, 2])
        X[n_samples // 4:n_samples // 2] += np.array([-2, -2])
        X[n_samples // 2:3 * n_samples // 4] += np.array([2, -2])
        X[3 * n_samples // 4:] += np.array([-2, 2])
        y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples // 2)])

    elif dataset_type == 'circle':
        X = np.random.randn(n_samples, 2) * 2
        radius = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
        y = (radius > 2.0).astype(int)

    if noise_prob > 0:
        flip_mask = np.random.rand(n_samples) < noise_prob
        y[flip_mask] = 1 - y[flip_mask]

    return X, y