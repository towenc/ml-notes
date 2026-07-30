import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regularization

    Regularization reduces the magnitude of the weights in order to reduce overfitting.

    ### Ridge Regression (L2)
    """)
    return


@app.cell
def _(lamw, np):
    def compute_cost(X, y, w, lam=0.0, lam2=0.0, reg=None) -> float:
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
        lam: float, default 0.0
            Regularization strength. Should be between 0 and 1.
            For elastic this is L1 strength
        lam2: float, default 0.0
            Only used in elastic. Should be between 0 and 1
            For elastic this is L2 strength
        reg: {None, "L2", "ridge", "L1", "lasso", "elastic"}, default None
            Which regularization penalty to add to the cost:
                - None           : no penalty (plain MSE)
                - "L2" / "ridge" : (lam/2m) * (w)^2
                - "L1" / "lasso" : (lam/m) * |w|
                - "elastic"      : (lam2/2m) * (w)^2 + (lam/m) * |w|

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
            J += (lam / (2*m)) * np.sum(w_reg**2)
        
        elif reg == "L1" or reg == "lasso":
            J += (lam / m) * np.sum(np.abs(w_reg))
        
        elif reg == "elastic":
            J += (lamw / (2*m)) * np.sum(w_reg**2) + (lam / m) * np.sum(np.abs(w_reg))
        return J               

    return


if __name__ == "__main__":
    app.run()
