import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    from sklearn.datasets import load_wine
    from sklearn.decomposition import PCA as SKPCA


    return SKPCA, go, load_wine, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Principal Component Analysis (PCA)
    A method for reducing dimensionality of data. It finds a new coordinate system for the data that maximises the variance in the data.

    ## Eigenvectures and Eigenvalues of the Covariance Matrix

    Compute the mean row. (average over features)
    $$\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i$$

    Subtract the mean from the data to get the zero mean data.
    $$X_C = X - \bar{x}$$

    The covariance matrix of $X_c$ is
    $$C = \frac{1}{n-1} X^{T}_C X_C$$

    The variance of the data projected onto a unit vector $w$ is
    $$Var(X_c w) = w^TCw$$

    For PCA we want to maximise the variance subject to $||w|| = 1$
    $$\max_{w} w^TCw \qquad subject\space to \qquad ||w|| = 1$$

    For solving constrained optimization problems we use the Langrange multipler, which simplies to the following (equation to find eigenvectors & values)
    $$Cw = \lambda w$$

    The principal components are therefore the eigenvectors of $C$ and the variance along each one is the eigenvalue. The final transformation where $Z$ is the data in the new coordiates and $\hat{X}$ is the reconstruction
    $$Z = X_c W_k \qquad \hat{X} = ZW_k^T + \bar{x}$$

    To find the fraction of variance explained by each prinicapl component:
    $$ variance \space ratio = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

    ## Singular Value Decomposition (SVD)
    SVD says that any matrix can factor as:
    $$A = U\Sigma V^\top$$

    $$
    A
    =
    \underbrace{
    \begin{bmatrix}
    u_{11} & u_{12} & \cdots & u_{1r} \\
    u_{21} & u_{22} & \cdots & u_{2r} \\
    \vdots & \vdots & \ddots & \vdots \\
    u_{n1} & u_{n2} & \cdots & u_{nr}
    \end{bmatrix}
    }_{U\;(n\times r)}
    \underbrace{
    \begin{bmatrix}
    \sigma_1 & 0 & \cdots & 0 \\
    0 & \sigma_2 & \cdots & 0 \\
    \vdots & \vdots & \ddots & \vdots \\
    0 & 0 & \cdots & \sigma_r
    \end{bmatrix}
    }_{\Sigma\;(r\times r)}
    \underbrace{
    \begin{bmatrix}
    v_{11} & v_{21} & \cdots & v_{d1} \\
    v_{12} & v_{22} & \cdots & v_{d2} \\
    \vdots & \vdots & \ddots & \vdots \\
    v_{1r} & v_{2r} & \cdots & v_{dr}
    \end{bmatrix}
    }_{V^{\mathsf{T}}\;(r\times d)}$$

    where:

    $U$ is the left-singular vectors. It is orthognal ($UU^\top = U^\top U = I$)

    $\Sigma$ is the singular values. A diagonal matrix.

    $V$ is the right-singular vectors. It is orthognal ($VV^\top = V^\top V = I$)

    Performing SVD on the covariance matrix gives:

    $$C = \frac{1}{n-1} X_C^\top \, X_C$$

    $$C = \frac{1}{n-1} V \Sigma U^\top U \Sigma V^\top$$

    Because $U$ is orthongal $U^\top U = 1$, leaving:

    $$C = V \frac{\Sigma^2}{n-1}V^\top$$

    This form is the same as the eigendecomposition of C ($Cw = \lambda w$), giving:

    $$W=V \qquad \lambda_i = \frac{s_i^2}{n-1}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementation
    """)
    return


@app.cell
def _(np):
    class PCA:
        """Principal Component Analysis via eigendecomposition of the covariance matrix.

        Parameters
        -----------
        mode : {"svd", "eig"}, default="svd"
            Selects the method to find the eigenvectors. "eig" uses np.linalg.eigh and "svd" uses singular value 
            decomposition using np.linalg.svd
        n_components : int or None, default=None
            Number of components to keep. None keeps all d.

        Attributes
        -----------
        mean_ : ndarray of shape (n_features,)
            Per-feature mean of the training data
        components_ : ndarray of shape (n_components, n_features)
            Principal axes, one per row, ordered by decreasing explained variance.
        explained_variance_ : ndarray of shape (n_components,)
            Variance along each component (the eigenvalues of the covariance matrix).
        explained_variance_ratio_ : ndarray of shape (n_components,)
            Fraction of the *total* variance carried by each component.
        """

        def __init__(self, mode="svd", n_components=None):
            self.n_components = n_components
            self.mode = mode

        def fit(self, X):
            """Finds the principle components of the data X

            Parameters
            -----------
            X : ndarray of shape (n_samples, n_features)

            Returns
            --------
            self
            """

            n = X.shape[0]
        
            # Perform centring of the data
            self.mean_ = np.mean(X, axis=0)
            Xc = X - self.mean_

            if self.mode == "eig":
                # Compute the covariance matrix
                C = (Xc.T @ Xc) / (n - 1)
    
                # Find the eigenvalues and eigenvectors of the covariance matrix
                eigvals, eigvecs = np.linalg.eigh(C)

                # Sorts in descending order
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
            elif self.mode == "svd":
                # SVD on Xc
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            
                eigvecs = Vt.T
                eigvals = (S ** 2) / (n - 1)
            else:
                raise ValueError(f"{self.mode} is invalid.")

            top_k_eigvals = eigvals[:self.n_components]
            top_k_eigvecs = eigvecs[:, :self.n_components]

            top_k_eigvecs = top_k_eigvecs

            self.components_ = top_k_eigvecs.T
            self.explained_variance_ = top_k_eigvals

            self.explained_variance_ratio_ = top_k_eigvals / np.sum(eigvals)

            return self

        def transform(self, X):
            """Project X onto the principal axes.

            Returns
            --------
            Z : ndarray of shape (n_samples, n_components)
            """
            Xc = X - self.mean_
            return Xc @ self.components_.T

        def fit_transform(self, X):
            return self.fit(X).transform(X)

        def inverse_transform(self, Z):
            """Map projected data back into the original feature space.

            Returns
            --------
            X_hat : ndarray of shape (n_samples, n_features)
            """
            return Z @ self.components_ + self.mean_ 

    return (PCA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison
    """)
    return


@app.cell
def _(load_wine):
    wine = load_wine()
    X_raw, y, names, classes = wine.data, wine.target, list(wine.feature_names), wine.target_names

    # Scale features
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0, ddof=1)
    return X, classes, y


@app.cell
def _(PCA, SKPCA, X, mo, np):
    pca    = PCA(n_components=3).fit(X)
    pca_sk = SKPCA(n_components=3).fit(X)

    Z    = pca.transform(X)
    Z_sk = pca_sk.transform(X)

    mo.md(f"""
    from scratch
    -------------
    **Explained variance:** {np.round(pca.explained_variance_, 2)}\n
    **Explained variance ratio:** {np.round(pca.explained_variance_ratio_, 2)}

    Sklearn
    --------
    **Explained variance:** {np.round(pca_sk.explained_variance_, 2)} \n
    **Explained variance ratio:** {np.round(pca_sk.explained_variance_ratio_, 2)}

    Variance explained matches between sklearn and from scratch implementation.
    """  
    )
    return Z, pca


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualisation
    """)
    return


@app.cell
def _(Z, classes, go, pca, y):
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    _ratios = pca.explained_variance_ratio_

    fig_3d = go.Figure()
    for _k in range(3):
        _m = y == _k
        fig_3d.add_trace(go.Scatter3d(
            x=Z[_m, 0], y=Z[_m, 1], z=Z[_m, 2],
            mode="markers",
            name=str(classes[_k]),
            marker=dict(size=4, color=colors[_k], opacity=0.85,
                        line=dict(width=0.5, color="white")),
        ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({_ratios[0]:.1%})",
            yaxis_title=f"PC2 ({_ratios[1]:.1%})",
            zaxis_title=f"PC3 ({_ratios[2]:.1%})",
            aspectmode="cube",
        ),
        title=f"Wine with 3 PC's - {_ratios.sum():.1%} of total variance",
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
    )
    fig_3d
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deciding on no. of Components
    Use a Skree plot to decide based on how many compoenents you need for an adequate level of variance explained
    """)
    return


@app.cell
def _(PCA, X, colors, np, plt):
    _full = PCA().fit(X)
    _ks = np.arange(1, _full.explained_variance_ratio_.size + 1)
    _cum = np.cumsum(_full.explained_variance_ratio_)

    # Measure the loss directly rather than trusting the ratios.
    _total_var = X.var(axis=0, ddof=1).sum()
    _measured = []
    for _k in _ks:
        _p = PCA(n_components=_k).fit(X)
        _measured.append(((X - _p.inverse_transform(_p.transform(X))) ** 2).sum() / (len(X) - 1) / _total_var)

    fig_scree, axes_scree = plt.subplots(1, 2, figsize=(11, 4))

    axes_scree[0].bar(_ks, _full.explained_variance_ratio_, color=colors[0], width=0.6)
    axes_scree[0].plot(_ks, _cum, color=colors[1], marker="o", label="cumulative")
    axes_scree[0].set_xticks(_ks)
    axes_scree[0].set_xlabel("component")
    axes_scree[0].set_ylabel("fraction of total variance")
    axes_scree[0].set_title("explained variance")
    axes_scree[0].legend()

    axes_scree[1].plot(_ks, 1 - _cum, color=colors[1], lw=4, alpha=0.85, label="predicted")
    axes_scree[1].plot(_ks, _measured, color=colors[0], marker="o", lw=0, mfc="white", label="measured")
    axes_scree[1].set_xticks(_ks)
    axes_scree[1].set_xlabel("components kept (k)")
    axes_scree[1].set_ylabel("fraction of variance lost")
    axes_scree[1].set_title("reconstruction error")
    axes_scree[1].legend()

    fig_scree.tight_layout()
    fig_scree
    return


if __name__ == "__main__":
    app.run()
