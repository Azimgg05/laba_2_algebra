import numpy as np
import matplotlib.pyplot as plt

# Импорты из модулей
from data_utils import get_standard_data, generate_custom_data
from models import SingleLayerPerceptron, AdvancedPerceptron
from metrics import calculate_metrics, plot_errors, k_fold_cross_validation


def plot_decision_boundary(model, X, y, title, filename=None):
    """Визуализация разделяющей границы и точек данных с возможностью сохранения."""
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
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


def plot_loss_curves(losses_dict, title="Динамика функции потерь", filename=None):
    """Универсальная функция для построения и сохранения графиков Loss."""
    plt.figure(figsize=(8, 6))
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label, linewidth=2)
    plt.title(title)
    plt.xlabel('Эпоха')
    plt.ylabel('Значение Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


def main():
    print("==================================================")
    print("--- 1. ОБЯЗАТЕЛЬНАЯ ЧАСТЬ ---")
    print("==================================================")
    X_train, X_test, y_train, y_test, X_full = get_standard_data()

    base_model = SingleLayerPerceptron(n_features=2)
    base_model.fit(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32)

    print(f"Точность на обучении (Train Accuracy): {np.mean(base_model.predict(X_train) == y_train):.4f}")
    print(f"Точность на тесте (Test Accuracy): {np.mean(base_model.predict(X_test) == y_test):.4f}")

    plot_loss_curves({
        'Train Loss': base_model.train_losses,
        'Val Loss': base_model.val_losses
    }, "Обязательная часть: Динамика Loss", filename='loss_plot.png')

    plot_decision_boundary(base_model, X_train, y_train, "Разделяющая граница (Базовый перцептрон)",
                           filename='decision_boundary_base.png')

    print("\n==================================================")
    print("--- ЗАДАНИЕ 3. МЕТРИКИ КАЧЕСТВА И АНАЛИЗ ОШИБОК ---")
    print("==================================================")
    y_pred = base_model.predict(X_test)
    y_prob = base_model.forward(X_test)

    metrics = calculate_metrics(y_test, y_pred, y_prob=y_prob, filename='roc_curve.png')
    print("Метрики качества на тестовой выборке:")
    for metric_name, val in metrics.items():
        print(f"  {metric_name}: {val:.4f}")

    plot_errors(X_test, y_test, y_pred, filename='error_analysis.png')

    print("\n==================================================")
    print("--- ЗАДАНИЕ 1. СОБСТВЕННЫЙ ГЕНЕРАТОР ДАННЫХ ---")
    print("==================================================")
    datasets = ['linear', 'circle', 'xor']
    for ds_type in datasets:
        print(f"Генерация и обучение на датасете: {ds_type.upper()}")
        X_custom, y_custom = generate_custom_data(dataset_type=ds_type, noise_prob=0.05)

        model_custom = SingleLayerPerceptron(n_features=2)
        model_custom.fit(X_custom, y_custom, X_custom, y_custom, epochs=100, lr=0.1, batch_size=32)

        plot_decision_boundary(model_custom, X_custom, y_custom, f"Генератор данных: {ds_type.upper()}",
                               filename=f'custom_boundary_{ds_type}.png')

    print("\n==================================================")
    print("--- ЗАДАНИЕ 2. ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ПОТЕРЬ И РЕГУЛЯРИЗАЦИЯ ---")
    print("==================================================")
    model_hinge = AdvancedPerceptron(n_features=2, loss_type='hinge')
    model_hinge.fit(X_train, y_train, epochs=100, lr=0.01, batch_size=32)

    model_l2 = AdvancedPerceptron(n_features=2, loss_type='bce', l2_lambda=0.1)
    model_l2.fit(X_train, y_train, epochs=100, lr=0.1, batch_size=32)

    plot_loss_curves({
        'Hinge Loss': model_hinge.losses,
        'BCE + L2 (lambda=0.1)': model_l2.losses
    }, "Задание 2: Сравнение сходимости функций потерь", filename='loss_comparison_hinge_l2.png')

    plot_decision_boundary(model_hinge, X_train, y_train, "Разделяющая граница: Hinge Loss",
                           filename='decision_boundary_hinge.png')

    print("\nИсследование влияния L2-регуляризации на значения весов:")
    l2_lambdas = [0.0, 0.1, 1.0, 10.0]
    for l2_val in l2_lambdas:
        model_l2_test = AdvancedPerceptron(n_features=2, loss_type='bce', l2_lambda=l2_val)
        model_l2_test.fit(X_train, y_train, epochs=100, lr=0.1, batch_size=32)
        weights_str = np.round(model_l2_test.w, 4)
        print(f"  Lambda = {l2_val:<4} | Веса модели w: {weights_str}")

    print("\n==================================================")
    print("--- ЗАДАНИЕ 4. ИССЛЕДОВАНИЕ СХОДИМОСТИ (MOMENTUM) ---")
    print("==================================================")
    momentum_betas = [0.0, 0.5, 0.9, 0.99]
    momentum_losses = {}

    print("Запуск обучения с различными значениями импульса (beta)...")
    for beta in momentum_betas:
        model_mom = AdvancedPerceptron(n_features=2, loss_type='bce', momentum_beta=beta)
        model_mom.fit(X_train, y_train, epochs=50, lr=0.01, batch_size=16)
        momentum_losses[f'Beta = {beta}'] = model_mom.losses

    plot_loss_curves(momentum_losses, "Задание 4: Влияние Momentum на скорость сходимости",
                     filename='momentum_loss_curves.png')

    print("\n==================================================")
    print("--- ЗАДАНИЕ 5. КРОСС-ВАЛИДАЦИЯ И ФИНАЛЬНАЯ МОДЕЛЬ ---")
    print("==================================================")
    best_params = k_fold_cross_validation(X_train, y_train, k=5)

    print("\nОбучение финальной модели на полном объеме обучающих данных...")
    final_model = AdvancedPerceptron(n_features=2, loss_type='bce')
    final_model.fit(X_train, y_train, epochs=100,
                    lr=best_params['lr'],
                    batch_size=best_params['batch_size'])

    final_acc = np.mean(final_model.predict(X_test) == y_test)
    print(f"Итоговая точность финальной модели на тестовой выборке: {final_acc:.4f}")

    plot_decision_boundary(final_model, X_test, y_test, "Финальная модель (После кросс-валидации)",
                           filename='decision_boundary_final.png')
    print("==================================================")


if __name__ == "__main__":
    main()