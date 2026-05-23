import numpy as np


class SingleLayerPerceptron:
    """Базовый перцептрон для обязательной части работы."""

    def __init__(self, n_features, init_type='small_random'):
        if init_type == 'zero':
            self.w = np.zeros(n_features)
        elif init_type == 'large':
            self.w = np.random.normal(0, 10, n_features)
        else:
            self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32):
        self.train_losses = []
        self.val_losses = []
        n_samples = X_train.shape[0]

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                y_pred = self.forward(X_batch)
                dw = np.dot(X_batch.T, (y_pred - y_batch)) / X_batch.shape[0]
                db = np.mean(y_pred - y_batch)

                self.w -= lr * dw
                self.b -= lr * db

            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)


class AdvancedPerceptron:
    """Продвинутый перцептрон с L2-регуляризацией, Momentum и Hinge Loss."""

    def __init__(self, n_features, loss_type='bce', l2_lambda=0.0, momentum_beta=0.0):
        self.w = np.random.randn(n_features) * 0.01
        self.b = 0.0
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda
        self.beta = momentum_beta

        self.v_w = np.zeros(n_features)
        self.v_b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        if self.loss_type == 'hinge':
            return z
        return self.sigmoid(z)

    def compute_loss(self, X, y):
        pred = self.forward(X)
        l2_penalty = (self.l2_lambda / 2) * np.sum(self.w ** 2)

        if self.loss_type == 'hinge':
            y_hinge = np.where(y == 0, -1, 1)
            loss = np.mean(np.maximum(0, 1 - y_hinge * pred))
        else:
            epsilon = 1e-15
            pred = np.clip(pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

        return loss + l2_penalty

    def fit(self, X, y, epochs=100, lr=0.1, batch_size=32):
        self.losses = []
        n_samples = X.shape[0]
        y_hinge = np.where(y == 0, -1, 1)

        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in range(0, n_samples, batch_size):
                idx = indices[i:i + batch_size]
                X_b, y_b, y_h_b = X[idx], y[idx], y_hinge[idx]

                if self.loss_type == 'hinge':
                    z = np.dot(X_b, self.w) + self.b
                    margin = y_h_b * z
                    dw = np.zeros_like(self.w)
                    db = 0.0
                    for j in range(len(X_b)):
                        if margin[j] < 1:
                            dw -= y_h_b[j] * X_b[j]
                            db -= y_h_b[j]
                    dw = dw / len(X_b) + self.l2_lambda * self.w
                    db = db / len(X_b)
                else:
                    pred = self.sigmoid(np.dot(X_b, self.w) + self.b)
                    dw = np.dot(X_b.T, (pred - y_b)) / len(X_b) + self.l2_lambda * self.w
                    db = np.mean(pred - y_b)

                self.v_w = self.beta * self.v_w + lr * dw
                self.v_b = self.beta * self.v_b + lr * db
                self.w -= self.v_w
                self.b -= self.v_b

            self.losses.append(self.compute_loss(X, y))

    def predict(self, X):
        if self.loss_type == 'hinge':
            return (np.dot(X, self.w) + self.b >= 0).astype(int)
        return (self.forward(X) >= 0.5).astype(int)