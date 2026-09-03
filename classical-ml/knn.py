import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_diabetes, load_iris, make_circles, make_moons
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.model_selection import train_test_split

    return KNeighborsClassifier, make_moons, mo, mpl, np, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #k-Nearest Neighbours (kNN)

    The kNN algorithm is as follows:
    For a given query point $x$:
    1. Compute distance from $x$ from to every n training point.
    2. Select the k smallest
    3. For classification, return the majority class. For regression return the mean.

    The distance used can be calculated in different ways.
    Considering n dimensional space and vectors p and q.

    **Eucliean**
    $$d = \sqrt{\sum_{i=1}^{n}(p_i - q_i )^2}$$
    **Manhattan**
    $$d = |\sum_{i=1}^{n}p_i - q_i |$$
    **Minkowski**
    $$d = \sqrt[p]{\sum_{i=1}^{n}(p_i - q_i )^p}$$
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
    class kNN:

        def __init__(self, k = 5, mode="classification", p=2):
            self.mode = mode
            self.p = p
            self.k = k

        def _compute_distances(self, x, X):

            distances = (np.sum(abs((X - x))**self.p, axis = 1)) ** (1/self.p)
          
            return distances

        def _kneighbours(self, x):
            distances = self._compute_distances(x, self.X_train_)
            idx = np.argpartition(distances, self.k)[: self.k]
            idx = idx[np.argsort(distances[idx])]
            return idx

        def _aggregate(self, labels):

            if self.mode == "regression":
                return np.mean(labels)
            elif self.mode == "classification":

                values, counts = np.unique(labels, return_counts=True)
                return values[np.argmax(counts)]
            else:
                raise ValueError(f"Unknown mode {self.mode}")

        def fit(self, X, y):
            self.X_train_ = X
            self.y_train_  = y

            return self

        def predict(self, X):

            predictions = []

            for x in X:
                idx = self._kneighbours(x)
                labels = self.y_train_[idx]
                predictions.append(self._aggregate(labels))

            return np.array(predictions)
            
        
        


    return (kNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison
    """)
    return


@app.cell
def _(make_moons, plt):
    # Two interleaving crescents. No straight line separates them - which is the
    # point: a linear model can't do this and kNN needs nothing extra to manage it.
    X, y = make_moons(n_samples=400, noise=0.15, random_state=0)
    names = ["feature 1", "feature 2"]
    classes = ["class 0", "class 1"]

    # Plot
    fig, ax = plt.subplots(figsize=(6,5))
    colors = ["#4C72B0", "#DD8452"]
    for k in range(2):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], c=colors[k], label=classes[k],
                   s=35, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
    ax.legend()

    return X, classes, names, y


@app.cell
def _(KNeighborsClassifier, X, kNN, mo, train_test_split, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    # from scratch
    knn_c = kNN().fit(Xtr, ytr)

    sk_knn_c = KNeighborsClassifier().fit(Xtr, ytr)

    tr_acc_mine = (knn_c.predict(Xtr) == ytr).mean()
    tr_acc_sk  = (sk_knn_c.predict(Xtr) == ytr).mean()

    te_acc_mine = (knn_c.predict(Xte) == yte).mean()
    te_acc_sk   = (sk_knn_c.predict(Xte) == yte).mean()

    mo.md(
        f"""
        **from scratch**
        ----------------
        Training Accuracy: {tr_acc_mine:.4f} \n
        Test Accuracy:     {te_acc_mine:.4f}

        **sklearn**
        ----------------
        Training Accuracy: {tr_acc_sk:.4f} \n
        Test Accuracy:     {te_acc_sk:.4f} \n

        Accuraries match between sklearn and from scratch implementation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision boundary
    """)
    return


@app.cell
def _(mo, mpl, np):
    def viz_palette():
        """Reference categorical slots 1-3, stepped for the notebook's theme."""
        _dark = mo.app_meta().theme == "dark"
        return {
            "series": ["#3987e5", "#d95926", "#199e70"] if _dark else ["#2a78d6", "#eb6834", "#1baf7a"],
            "surface": "#1a1a19" if _dark else "#fcfcfb",
            "ink": "#ffffff" if _dark else "#0b0b0b",
            "ink2": "#c3c2b7" if _dark else "#52514e",
        }


    def plot_decision_boundary(ax, model, X, y, class_names=None, feature_names=None,
                               resolution=200, title=None, legend=True):
        """Fill the plane with the model's prediction, then draw the training points.

        Takes any fitted estimator with a .predict, so ours and sklearn's can be
        drawn on adjacent axes. Returns the grid of predictions so callers can
        compare two models cell by cell.
        """
        _pal = viz_palette()
        _series, _surface = _pal["series"], _pal["surface"]
        _ink, _ink2 = _pal["ink"], _pal["ink2"]
        _names = list(class_names) if class_names is not None else [f"class {c}" for c in np.unique(y)]

        def _tint(hex_colour, keep):
            _c = np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]) / 255
            _s = np.array([int(_surface[i:i + 2], 16) for i in (1, 3, 5)]) / 255
            return tuple(_c * keep + _s * (1 - keep))

        _pad = 0.5
        _xs = np.linspace(X[:, 0].min() - _pad, X[:, 0].max() + _pad, resolution)
        _ys = np.linspace(X[:, 1].min() - _pad, X[:, 1].max() + _pad, resolution)
        _gx, _gy = np.meshgrid(_xs, _ys)
        _zz = model.predict(np.column_stack([_gx.ravel(), _gy.ravel()])).reshape(_gx.shape).astype(float)

        ax.set_facecolor(_surface)
        # regions kept faint so the training points stay the foreground
        ax.pcolormesh(
            _gx, _gy, _zz, shading="nearest",
            cmap=mpl.colors.ListedColormap([_tint(c, 0.20) for c in _series]),
            vmin=-0.5, vmax=len(_series) - 0.5,
        )
        ax.contour(_gx, _gy, _zz, levels=[0.5, 1.5], colors=[_ink2], linewidths=1.0, alpha=0.5)

        # a surface-coloured ring keeps overlapping points countable
        for _i, _cls in enumerate(np.unique(y)):
            _m = y == _cls
            ax.scatter(X[_m, 0], X[_m, 1], s=30, c=_series[_i], edgecolors=_surface,
                       linewidths=1.2, zorder=3, label=_names[_i])

        ax.set_xlim(_xs[0], _xs[-1])
        ax.set_ylim(_ys[0], _ys[-1])
        for _side in ("top", "right"):
            ax.spines[_side].set_visible(False)
        for _side in ("left", "bottom"):
            ax.spines[_side].set_color(_ink2)
            ax.spines[_side].set_linewidth(0.8)
        ax.tick_params(colors=_ink2, labelsize=8, length=3)
        if feature_names is not None:
            ax.set_xlabel(feature_names[0], color=_ink2, fontsize=9)
            ax.set_ylabel(feature_names[1], color=_ink2, fontsize=9)
        if title:
            ax.set_title(title, color=_ink, fontsize=10.5, pad=8, loc="left")
        if legend:
            _leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5, handletextpad=0.4)
            for _t in _leg.get_texts():
                _t.set_color(_ink)
        return _zz


    return plot_decision_boundary, viz_palette


@app.cell
def _(
    KNeighborsClassifier,
    X,
    classes,
    kNN,
    names,
    plot_decision_boundary,
    plt,
    viz_palette,
    y,
):
    # Same k, same data, same metric - the only difference is whose code drew it.
    _k = 5
    _pal = viz_palette()

    _fig, _axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=140, sharex=True, sharey=True)
    _fig.patch.set_facecolor(_pal["surface"])

    _zz_mine = plot_decision_boundary(
        _axes[0], kNN(k=_k).fit(X, y), X, y,
        class_names=classes, feature_names=names, title="from scratch", legend=True,
    )
    _zz_sk = plot_decision_boundary(
        _axes[1], KNeighborsClassifier(n_neighbors=_k).fit(X, y), X, y,
        class_names=classes, feature_names=names, title="sklearn KNeighborsClassifier", legend=False,
    )
    _axes[1].set_ylabel("")

    _disagree = (_zz_mine != _zz_sk).sum()
    _fig.suptitle(
        f"kNN decision boundary, k={_k}, p=2   -   grid cells where the two disagree: "
        f"{_disagree} of {_zz_mine.size:,}",
        color=_pal["ink"], fontsize=11.5, x=0.008, ha="left", y=0.99,
    )
    _fig.tight_layout(rect=[0, 0, 1, 0.94])
    _fig

    return


if __name__ == "__main__":
    app.run()
