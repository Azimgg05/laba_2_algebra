import numpy as np
import matplotlib.pyplot as plt
from models import AdvancedPerceptron


def calculate_metrics(y_true, y_pred, y_prob=None, show_roc=True):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {'Precision': precision, 'Recall': recall, 'F1': f1}

    if y_prob is not None:
        thresholds = np.sort(np.unique(y_prob))[::-1]
        tpr_list, fpr_list = [0], [0]

        for t in thresholds:
            y_p = (y_prob >= t).astype(int)
            TP_t = np.sum((y_true == 1) & (y_p == 1))
            FP_t = np.sum((y_true == 0) & (y_p == 1))

            tpr_list.append(TP_t / np.sum(y_true == 1))
            fpr_list.append(FP_t / np.sum(y_true == 0))

        tpr_list.append(1)
        fpr_list.append(1)

        auc = np.trapezoid(tpr_list, fpr_list)
        metrics['ROC-AUC'] = auc

        if show_roc:
            plt.figure(figsize=(6, 6))
            plt.plot(fpr_list, tpr_list, marker='.', label=f'AUC = {auc:.3f}')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.title('ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.grid()
            plt.show()

    return metrics


def plot_errors(X, y_true, y_pred):
    errors = y_true != y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(X[~errors, 0], X[~errors, 1], c=y_true[~errors], cmap='bwr', alpha=0.5, label='Верно')
    plt.scatter(X[errors, 0], X[errors, 1], c='lime', marker='x', s=100, label='Ошибки')
    plt.title('Анализ ошибок классификации')
    plt.legend()
    plt.show()


def k_fold_cross_validation(X, y, k=5, lr_list=[0.01, 0.1], batch_list=[16, 32]):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_sizes = len(X) // k

    best_acc = 0
    best_params = {}

    print(f"Запуск {k}-кратной кросс-валидации...")
    for lr in lr_list:
        for bs in batch_list:
            fold_accuracies = []

            for i in range(k):
                val_idx = indices[i * fold_sizes: (i + 1) * fold_sizes]
                train_idx = np.concatenate([indices[:i * fold_sizes], indices[(i + 1) * fold_sizes:]])

                X_tr, y_tr = X[train_idx], y[train_idx]
                X_v, y_v = X[val_idx], y[val_idx]

                model = AdvancedPerceptron(n_features=X.shape[1])
                model.fit(X_tr, y_tr, epochs=50, lr=lr, batch_size=bs)

                acc = np.mean(model.predict(X_v) == y_v)
                fold_accuracies.append(acc)

            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            print(f"LR: {lr}, Batch: {bs} | Точность: {mean_acc:.4f} ± {std_acc:.4f}")

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = {'lr': lr, 'batch_size': bs}

    print(f"Лучшие параметры: {best_params} (Точность: {best_acc:.4f})")
    return best_params