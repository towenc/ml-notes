import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

    return (
        ElasticNet,
        Lasso,
        LinearRegression,
        PolynomialFeatures,
        Ridge,
        StandardScaler,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization Implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Regularization reduces the magnitude of the weights in order to reduce overfitting.
    It does this by penalises the magnitude of weights in the cost function.
    Notice that the weight vector in the penalty term $\tilde{w}$ zeros out the intercept. The bias is usually not regularized.
    ### Ridge Regression (L2)
    $$
    J(w) = \frac{1}{2m}\lVert Xw - y \rVert^2 + \frac{\lambda}{2m}\lVert \tilde{w} \rVert^2
    \qquad
    \nabla J(w) = \frac{1}{m}X^\top(Xw - y) + \frac{\lambda}{m}\tilde{w}
    $$

    $$ where \space \tilde{w}_j = \begin{cases} 0 & j = 0 \\ w_j & j \geq 1 \end{cases} $$
    For ridge regression, the weights are shrunk but do not go to zero. The cost function is convex and smooth so regular gradient descent works.


    ### Lasso Regression (L1)
    $$
    J(w) = \frac{1}{2m}\lVert Xw - y \rVert^2 + \frac{\lambda}{m}\lVert \tilde{w} \rVert_1
    \qquad
    \nabla J(w) = \frac{1}{m}X^\top(Xw - y) + \frac{\lambda}{m}\operatorname{sign}(\tilde{w})
    $$
    For lasso regression, the weights can be reduced to zero. The cost function is convex but is not smooth as there are non-differeniable "kinks" at 0.

    Proximal Gradient Descent (ISTA) is used instead. It performs normal gradient descent but then snaps the weight to 0 if its within a threshold to 0.


    ### Elastic Net
    $$
    J(w) = \frac{1}{2m}\lVert Xw - y \rVert^2 + \frac{\lambda_2}{2m}\lVert \tilde{w} \rVert^2 + \frac{\lambda_1}{m}\lVert \tilde{w} \rVert_1
    \qquad
    \nabla J(w) = \frac{1}{m}X^\top(Xw - y) + \frac{\lambda_2}{m}\tilde{w} + \frac{\lambda_1}{m}\operatorname{sign}(\tilde{w})
    $$

    Elastic Net is a combination of ridge and lasso regression. The balance between each is set by $\lambda_1$ and $\lambda_2$. Like lasso regression, ISTA is used.
    """)
    return


@app.cell
def _(np):
    def compute_cost(X, y, w, lam1=0.0, lam2=0.0, reg=None) -> float:
        """Compute the (optionally regularized) linear-regression cost.
        Base cost is J = (1/2m) * (Xw - y)^2.
        Regularization cost is added depending on reg

        Parameters
        -----------
        X: ndarray, shape (m, n)
            Feature matrix. Includes the intercept
        y: ndarray, shape (m, )
            Target values
        w: ndarray, shape (n, )
            Weight vector
        lam1: float, default 0.0
            L1 regularization strength.
            Used for elastic and lasso.
         lam2: float, default 0.0
             L2 regularization strength. 
             Used for elastic and ridge.
        reg: {None, "L2", "ridge", "L1", "lasso", "elastic"}, default None
            Which regularization penalty to add to the cost:
                - None           : no penalty (plain MSE)
                - "L2" / "ridge" : (lam2/2m) * (w)^2
                - "L1" / "lasso" : (lam1/m) * |w|
                - "elastic"      : (lam2/2m) * (w)^2 + (lam1/m) * |w|

        Returns
        -----------
        float
            The scalar cost J

        """
        m = X.shape[0]
        J = (1 / (2*m)) * np.sum((X @ w - y) ** 2)

        # removes the intercept so it is not regularized
        w_reg = w[1:]
        if reg == "L2" or reg == "ridge":
            J += (lam2 / (2*m)) * np.sum(w_reg**2)

        elif reg == "L1" or reg == "lasso":
            J += (lam1 / m) * np.sum(np.abs(w_reg))

        elif reg == "elastic":
            J += (lam2 / (2*m)) * np.sum(w_reg**2) + (lam1 / m) * np.sum(np.abs(w_reg))
        return J

    def compute_grad(X, y, w, lam2=0.0, reg=None) -> np.ndarray:
        """Compute the (optionally regularized) gradient for linear regression for GD
        Base gradient is dJ/dw = (1/m) * (Xw - y)X 
        Only L2 regularization for ridge and elastic are applied here.
        L1 regularization for lasso and ridge are applied in gradient_descent (ITSA)

        Parameters
        -----------
        X: ndarray, shape (m, n)
            Feature matrix. Includes the intercept
        y: ndarray, shape (m, )
            Target values
        w: ndarray, shape (n, )
            Weight vector
        lam2: float, default 0.0
            L2 regularization strength. 
            Used for elastic and ridge
        reg: {None, "L2", "ridge", "L1", "lasso", "elastic"}, default None
            Which regularization penalty to add to the cost:
                - None           : no penalty (plain MSE grad)
                - "L2" / "ridge" : (lam / m) * w
                - "L1" / "lasso" : (lam / m) * sign(w)
                - "elastic"      : lam2 / m) * w + (lam / m) * sign(w)

        Returns
        -----------
        ndarry
            Gradient vector, shape (n, )
        """
        m, n = X.shape
        grad = (1/m) * (X.T @ (X @ w - y))
    
        reg_grad = np.zeros(n)
        if reg in ("L2", "ridge", "elastic"):
            reg_grad = (lam2 / m) * w
 
        # drop the intercept. should not be included in regularization
        reg_grad[0] = 0
        grad += reg_grad
    
        return grad

    def soft_threshold(w, t):
        """
        Snaps weights to 0 if they approach 0 within the treshold t.
        Used for proximal gradient descent (ITSA)
        """
        return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)
    
    def gradient_descent(X, y, lr=0.1, lam1=0.0, lam2=0.0, reg=None, n_iter=1000):
        """Performs gradient descent for linear, lasso, ridge and elastic net regression.
        For lasso and elastic net proximal gradient descent is performed
        """
        m, n = X.shape
    
        w = np.zeros(n) # initalize weights to 0
        history = []

        for i in range(n_iter):
            grad = compute_grad(X, y, w, lam2=lam2, reg=reg)
            w -= lr*grad

            # Proximal Gradient Descent (ITSA)
            if reg in ("L1", "lasso", "elastic"):
                w[1:] = soft_threshold(w[1:], lr * lam1 /m)
            
            history.append(compute_cost(X, y, w, lam1=lam1, lam2=lam2, reg=reg))

        return w, history

    def predict(X, w):
        """
        Computes predictions using weights and data
        """
        yhat = X @ w

        return yhat

    def rmse(yhat, y):
        """
        Compute root mean squared error.
        """
        error = np.sqrt(np.mean((yhat - y) ** 2))
        return np.sum(error)

    return gradient_descent, predict, rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Data
    """)
    return


@app.cell
def _(PolynomialFeatures, StandardScaler, np, plt):
    # Create linear data
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-1, 1, 50))
    y_true = 3 * x + 1
    y = y_true + 2 * rng.standard_normal(50)

    # Create polynomial features up to degree 15
    poly= PolynomialFeatures(degree=7, include_bias=True)
    X = poly.fit_transform(x.reshape(-1, 1)) #reshape to x rows and 1 column

    # Scale the features
    scaler = StandardScaler()
    # Intercept does not get scaled
    X[:, 1:] = scaler.fit_transform(X[:, 1:])

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(x, y, color="black", label="data")
    x_line= np.linspace(-1, 1, 200)
    ax.plot(x_line, 2*x_line + 1, color="steelblue", linewidth=2, label="true trend (3x+1)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig
    return X, poly, scaler, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison with Sklearn
    """)
    return


@app.cell
def _(
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    X,
    gradient_descent,
    np,
    predict,
    rmse,
    y,
):
    # sklearn uses alpha instead of lambda.
    # alpha is the overall penalty strength and l1_ratio is how it splits between L1 and L2
    alpha, l1_ratio = 20, 0.5
    m = X.shape[0]

    gd_params = {
        "None":    dict(lam1=0.0,              lam2=alpha),
        "ridge":   dict(lam1=0.0,              lam2=alpha),
        "lasso":   dict(lam1=alpha,              lam2=0.0),
        "elastic": dict(lam1=alpha * l1_ratio,   lam2=alpha * (1 - l1_ratio)),
    }

    # sklearn scales penalties differently. 
    #   Lasso/ElasticNet  -> (1/2m)*MSE + alpha*penalty, so their alpha = ours / m
    #   Ridge             -> MSE + alpha*penalty with no 1/m, so their alpha = ours
    sk_models = {
        "None":  LinearRegression(fit_intercept=True),
        "ridge":   Ridge(alpha=alpha, fit_intercept=True),
        "lasso":   Lasso(alpha=alpha / m, fit_intercept=True),
        "elastic": ElasticNet(alpha=alpha / m, l1_ratio=l1_ratio, fit_intercept=True),
    }

    regs = ["None", "ridge", "lasso", "elastic"]

    gd_results = {}
    for reg in regs:
        w, _ = gradient_descent(X, y, reg=reg, n_iter=10000, **gd_params[reg])
        yhat = predict(X, w)
        gd_results[reg] = {"w": w, "yhat": yhat, "rmse": rmse(yhat, y)}

    sk_results = {}
    for reg, model in sk_models.items():
        model.fit(X, y)
        yhat = model.predict(X)
        w_sk = np.concatenate([[model.intercept_], model.coef_[1:]])
        sk_results[reg] = {"w": w_sk, "yhat": yhat, "rmse": rmse(yhat, y)}

    def fmt(v):
        return np.array2string(v, precision=4, suppress_small=True)

    for reg in regs:
        print(f"---- {reg} ----")
        print(f"GD Coefficients:       {fmt(gd_results[reg]['w'])}")
        print(f"SK Coefficients:       {fmt(sk_results[reg]['w'])}")
        print(f"GD RMSE:               {gd_results[reg]['rmse']:.6f}")
        print(f"SK RMSE:               {sk_results[reg]['rmse']:.6f}")
        print()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the results, we can see that "None" or linear regression has the smallest RMSE. This is because there is no regularization and it fits the training data better. However, this may translate to more variance when we go to predict on the test set.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing Regularization
    """)
    return


@app.cell
def _(
    X,
    gradient_descent,
    np,
    plt,
    poly,
    predict,
    rmse,
    scaler,
    viz_lam1,
    viz_lam2,
    viz_n_iter,
    x,
    y,
):
    _lam1 = viz_lam1.value
    _lam2 = viz_lam2.value

    # the penalty names are just labels for which lambdas are switched on
    if _lam1 > 0 and _lam2 > 0:
        _reg = "elastic"
    elif _lam1 > 0:
        _reg = "lasso"
    elif _lam2 > 0:
        _reg = "ridge"
    else:
        _reg = "None"

    # GD is stable only for lr < 2/L, and the L2 penalty raises L by lam2/m.
    # Without this, large lambda2 blows the weights up to NaN.
    _L = np.linalg.eigvalsh(X.T @ X / len(y))[-1] + _lam2 / len(y)
    _lr = min(0.1, 1.0 / _L)

    _w, _ = gradient_descent(
        X, y, reg=_reg, lam1=_lam1, lam2=_lam2, lr=_lr, n_iter=viz_n_iter.value
    )

    # put the smooth line through the same feature pipeline as X
    _xl = np.linspace(-1, 1, 200)
    _Xl = poly.transform(_xl.reshape(-1, 1))
    _Xl[:, 1:] = scaler.transform(_Xl[:, 1:])

    # fix the weight axis to the unpenalized solution so shrinkage is visible
    _bound = np.abs(np.linalg.lstsq(X, y, rcond=None)[0][1:]).max() * 1.1
    _terms = np.arange(1, len(_w))

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(11, 4.5))

    _a1.scatter(x, y, color="black", s=22, zorder=3, label="data")
    _a1.plot(_xl, 3 * _xl + 1, color="steelblue", lw=2, label="true trend (3x+1)")
    _a1.plot(_xl, _Xl @ _w, color="crimson", lw=2, label=f"fit ({_reg})")
    _a1.set_ylim(y.min() - 1.5, y.max() + 1.5)
    _a1.set_xlabel("x")
    _a1.set_ylabel("y")
    _a1.set_title(
        f"{_reg}   lambda1={_lam1:g}  lambda2={_lam2:g}"
        f"    train RMSE = {rmse(predict(X, _w), y):.3f}"
    )
    _a1.legend(loc="upper left", fontsize=8)

    _a2.bar(_terms, _w[1:], color="steelblue")
    _a2.axhline(0, color="black", lw=0.8)
    _a2.set_ylim(-_bound, _bound)
    _a2.set_xticks(_terms)
    _a2.set_xticklabels([f"$x^{d}$" for d in _terms])
    _a2.set_xlabel("polynomial term (intercept excluded)")
    _a2.set_ylabel("weight")
    _a2.set_title(f"{int(np.sum(_w[1:] == 0))} of {len(_w) - 1} weights exactly zero")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    viz_steps = [0, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]

    viz_lam1 = mo.ui.slider(steps=viz_steps, value=0, label="lambda1  (L1)", show_value=True)
    viz_lam2 = mo.ui.slider(steps=viz_steps, value=10, label="lambda2  (L2)", show_value=True)
    viz_n_iter = mo.ui.slider(
        start=1000, stop=20000, step=1000, value=5000, label="n_iter", show_value=True
    )

    mo.vstack([viz_lam1, viz_lam2, viz_n_iter])
    return viz_lam1, viz_lam2, viz_n_iter


if __name__ == "__main__":
    app.run()
