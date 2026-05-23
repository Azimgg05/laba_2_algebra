import numpy as np
import matplotlib.pyplot as plt

# Импорты из наших модулей
from data_utils import get_standard_data, generate_custom_data
from models import SingleLayerPerceptron, AdvancedPerceptron
from metrics import calculate_metrics, plot_errors, k_fold_cross_validation


def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='bwr')
    plt.title(title)
    plt.show()


def main():
    print("--- 1. Обязательная часть ---")
    X_train, X_test, y_train, y_test, X_full = get_standard_data()

    model = SingleLayerPerceptron(n_features=2)
    model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    print(f"Точность (Train): {np.mean(model.predict(X_train) == y_train):.4f}")
    print(f"Точность (Test): {np.mean(model.predict(X_test) == y_test):.4f}")

    plot_decision_boundary(model, X_train, y_train, "Разделяющая граница (базовый перцептрон)")

    print("\n--- 2. Метрики (Задание 3) ---")
    y_pred = model.predict(X_test)
    y_prob = model.forward(X_test)
    metrics = calculate_metrics(y_test, y_pred, y_prob=y_prob)
    print("Метрики на тестовой выборке:", metrics)
    plot_errors(X_test, y_test, y_pred)

    print("\n--- 3. Кросс-валидация (Задание 5) ---")
    best_params = k_fold_cross_validation(X_train, y_train, k=5)

    print("\n--- 4. Тест на XOR (Задание 1) ---")
    X_xor, y_xor = generate_custom_data(dataset_type='xor')
    model_xor = SingleLayerPerceptron(n_features=2)
    # Используем X_xor как валидационную выборку для простоты интерфейса fit
    model_xor.fit(X_xor, y_xor, X_xor, y_xor, epochs=100, lr=0.1)
    plot_decision_boundary(model_xor, X_xor, y_xor, "Попытка разделить XOR (Ожидаемый провал)")


if __name__ == "__main__":
    main()
