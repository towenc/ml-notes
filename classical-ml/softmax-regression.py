import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression, StandardScaler, make_blobs, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Softmax Regression
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Softmax regression is the more general form of logistic regression. It is used for multi-class classification whereas logistic regression is used for binary class classification.

    Softmax regression is defined by the following function:
    $$f_{\vec{w}_k, b_k}(\vec{x}) = g_k(\vec{z}) = \frac{e^{\vec{w}_k \cdot \vec{x} + b_k}}{\displaystyle\sum_{j=1}^{C} e^{\vec{w}_j \cdot \vec{x} + b_j}}$$

    The cost function is:
    $$J(W, \vec{b}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{C} y_k^{(i)} \log f_k(\vec{x}^{(i)})$$

    The gradient per class is:
    $$
    \frac{\partial J}{\partial \vec{w}_c} = \frac{1}{m} \sum_{i=1}^{m} \left( f_c(\vec{x}^{(i)}) - y_c^{(i)} \right) \vec{x}^{(i)}
    \qquad
    \frac{\partial J}{\partial b_c} = \frac{1}{m} \sum_{i=1}^{m} \left( f_c(\vec{x}^{(i)}) - y_c^{(i)} \right)
    $$

    In vectorized form:
    $$\nabla_W J = \frac{1}{m} X^\top (\hat{Y} - Y)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Note on Softmax Implementation
    The row max (largest logit per example) is subtracted from each row/data point.
    This is done to prevent overflow. $e^{big\space number}$ can be a very big number causing overflow.

    To prevent this the biggest logit is subtracted from each row. This does not change the calculation due to shift invariance.
    That is:
    $$
    \mathrm{softmax}(z_k - c)
    = \frac{e^{z_k - c}}{\sum_{j} e^{z_j - c}}
    = \frac{e^{z_k}\, e^{-c}}{\sum_{j} e^{z_j}\, e^{-c}}
    = \frac{e^{z_k}\, e^{-c}}{e^{-c} \sum_{j} e^{z_j}}
    = \frac{e^{z_k}}{\sum_{j} e^{z_j}}
    = \mathrm{softmax}(z_k)
    $$
    """)
    return


@app.cell
def _(np):
    def softmax(Z):
        """
        Applies softmax
        """
        Z = Z - np.max(Z, axis=1, keepdims=True) 
        f = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        return f

    def add_bias(X):
        """
        Adds a columns of ones to X to incorporate the bias
        """
        m = X.shape[0]
        bias = np.ones((m,1))
        X = np.hstack([bias, X])

        return X
    
    def predict_proba(X, w):
        """
        Predicts probabilites for softmax regression.
        X should include the bias
        """
        Z = X @ w 
        proba = softmax(Z)
        return proba

    def compute_cost(Y_hat, Y):
        """
        Computes cross-entropy loss and averages it over the data points to give the cost.
        1e-12 is added to Y_hat to prevent log(0)
        """
        m = Y.shape[0]
        L = -np.sum(Y * np.log(Y_hat + 1e-12), axis=1)
        J = (1/m) * np.sum(L, axis=0)

        return J

    def compute_grad(X, Y_hat, Y):
        """
        Computes the gradient for cost function of cross-entropy loss.
        Bias is included in X.
        """
        m = X.shape[0]
        grad = (1/m) * X.T @ (Y_hat - Y)
        return grad

    def gradient_descent(X, Y, lr=0.1, n_iters=1000):
        """
        Performs gradient descent for softmax regression.
        """
        m, C = Y.shape
        n = X.shape[1]
    
        w = np.zeros((n, C))
        history = []
        for i in range(n_iters):
            Y_hat = predict_proba(X, w)
            grad = compute_grad(X, Y_hat, Y)
            w -= lr * grad
            history.append(compute_cost(Y_hat, Y))

        return w, history
    

    return add_bias, gradient_descent, predict_proba


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison with Sklearn
    """)
    return


@app.cell
def _(StandardScaler, make_blobs, np, plt):
    # Create data
    cluster_std = 5.0
    X, y = make_blobs(n_samples=400, centers=4, n_features=2, cluster_std=cluster_std, random_state=42)
    X = StandardScaler().fit_transform(X)

    C = 4
    # One hot encode y
    Y = np.eye(C)[y]

    # plot
    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=40)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'4 blobs, cluster_std={cluster_std}')
    plt.colorbar(label='class')
    plt.show()
    return X, Y, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Linearly inseparable data is to be used. For linearly separable data, unregularized gradient descent for cross-entropy loss has no finite minimum. Since no regularization was implemented for GD implementation, this makes the comparison with Sklearn easier.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Fitting
    """)
    return


@app.cell
def _(
    LogisticRegression,
    X,
    Y,
    add_bias,
    gradient_descent,
    np,
    predict_proba,
    y,
):
    ## Gradient Descent
    Xb = add_bias(X)
    w, history = gradient_descent(Xb, Y, lr=0.1, n_iters=10000)
    P_gd = predict_proba(Xb, w)
    pred_gd = np.argmax(P_gd, axis = 1)

    gd_accuracy = (pred_gd == y).mean()

    ## Sklearn
    model = LogisticRegression(C=1e6, max_iter=10000) # C=1e6 means no regularization
    model.fit(X, y)
    P_sk = model.predict_proba(X)
    w_sk = np.vstack([model.intercept_, model.coef_.T])
    pred_sk = np.argmax(P_sk, axis = 1)

    sk_accuracy = (pred_sk == y).mean()

    # Report
    print("Comparison")
    print("=" * 38)
    print("GD weights")
    print(w)
    print("SK weights")
    print(w_sk)

    print(f"GD Accuracy = {gd_accuracy}")
    print(f"SK Accuracy = {sk_accuracy}")
    print(f"Predictions identical: {(pred_gd == pred_sk).all()}")
    print(f"No. of Predictions Agreeing  : {(pred_gd == pred_sk).mean():.4f}")
    print(f"Max |prob difference| : {np.abs(P_gd - P_sk).max():.4f}")
    return gd_accuracy, model, sk_accuracy, w


@app.cell
def _(
    X,
    add_bias,
    gd_accuracy,
    model,
    np,
    plt,
    predict_proba,
    sk_accuracy,
    w,
    y,
):
    # Plotting
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
 
    grid_gd = predict_proba(add_bias(grid), w).argmax(1).reshape(xx.shape)  # your model
    grid_sk = model.predict(grid).reshape(xx.shape)                        # sklearn
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, Z, title, acc in [(axes[0], grid_gd, "Gradient Descent (from scratch)", gd_accuracy),
                              (axes[1], grid_sk, "sklearn LogisticRegression", sk_accuracy)]:
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        ax.scatter(X[:,0], X[:,1], c=y, cmap='viridis', edgecolor='k', s=30)
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
        ax.set_title(f"{title}\naccuracy = {acc:.4f}")
    plt.suptitle('Softmax regression decision boundaries')
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
