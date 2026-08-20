import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np
    import matplotlib.pyplot as plt

    from numpy.random import default_rng
    from sklearn.datasets import load_diabetes, make_moons
    from sklearn.ensemble import GradientBoostingClassifier as SkGB_clf, GradientBoostingRegressor as SkGB_reg
    from sklearn.model_selection import train_test_split

    return (
        Optional,
        SkGB_clf,
        SkGB_reg,
        dataclass,
        default_rng,
        load_diabetes,
        make_moons,
        mo,
        np,
        plt,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gradient Boosting
    Gradient boosting fits trees sequentially. Every new tree is fit on the residuals of the previous tree.


    The algorithm for squared error regression is:

    1. Start with a constant prediction $F_0(x) = \text{mean}(y)$.
    2. For each round $m = 1 \dots M$:
        1. Compute the residuals $r_i = y_i - F_{m-1}(x_i)$.
        2. Fit a shallow regression tree $h_m$ to those residuals.
        3. Update $F_m(x) = F_{m-1}(x) + \eta \, h_m(x)$, where $\eta$ is the learning rate.

    The final prediction is $F_M(x)$: a constant, plus a sum of small tree-shaped corrections.

    The algorithm for classification:

    1. Start with a constant score $F_0(x) = \log\dfrac{\bar{y}}{1 - \bar{y}}$.
    2. For each round $m = 1 \dots M$:
        1. Compute probabilities $p_i = \sigma(F_{m-1}(x_i))$ and pseudo-residuals $r_i = y_i - p_i$.
        2. Fit a shallow regression tree to $(x_i, r_i)$.
        3. Replace each leaf value with $\gamma_j = \dfrac{\sum_{i \in R_j} r_i}{\sum_{i \in R_j} p_i(1 - p_i)}$.
        4. Update $F_m(x) = F_{m-1}(x) + \eta \sum_j \gamma_j \mathbb{1}(x \in R_{jm})$.
    3. Predict $\sigma(F_M(x))$, thresholding at 0.5 for the label.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Gradient boosting is good at **reducing bias**. As such, with more trees fit the greater the change of overfitting.
    """)
    return


@app.cell
def _(GradientBoostingRegressor, default_rng, np):
    _rng = default_rng(0)
    sigma1d = 0.5


    def truth1d(x):
        """The signal the model is trying to recover, without the noise."""
        return np.sin(x) + 0.15 * x

    # Generate small noisy data
    _x_tr = np.sort(_rng.uniform(0, 10, 40))
    X1d_tr = _x_tr.reshape(-1, 1)
    y1d_tr = truth1d(_x_tr) + _rng.normal(scale=sigma1d, size=_x_tr.size)

    _x_te = np.linspace(0, 10, 500)
    X1d_te = _x_te.reshape(-1, 1)
    y1d_te = truth1d(_x_te) + _rng.normal(scale=sigma1d, size=_x_te.size)

    gb1d = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=3
    ).fit(X1d_tr, y1d_tr)

    f"{len(gb1d.trees_)} trees fit on {len(X1d_tr)} noisy points (sigma = {sigma1d})"
    return X1d_te, X1d_tr, gb1d, truth1d, y1d_te, y1d_tr


@app.cell
def _(X1d_tr, gb1d, np, plt, truth1d, y1d_tr):
    _stages = [1, 5, 16, 300]
    _grid = np.linspace(0, 10, 400).reshape(-1, 1)

    _Fg = np.full(len(_grid), gb1d.F0_)
    _Ft = np.full(len(X1d_tr), gb1d.F0_)
    _snap = {0: (_Fg.copy(), _Ft.copy())}
    for _i, _tree in enumerate(gb1d.trees_, start=1):
        _Fg += gb1d.learning_rate * _tree.predict(_grid)
        _Ft += gb1d.learning_rate * _tree.predict(X1d_tr)
        _snap[_i] = (_Fg.copy(), _Ft.copy())

    _rlim = 1.1 * np.abs(y1d_tr - _snap[_stages[0]][1]).max()

    _fig, _axes = plt.subplots(2, len(_stages), figsize=(4.0 * len(_stages), 6.6),
                               sharex=True)
    for _j, _m in enumerate(_stages):
        _Fgrid, _Ftr = _snap[_m]

        _a = _axes[0, _j]
        _a.plot(_grid[:, 0], truth1d(_grid[:, 0]), c="#999", ls="--", lw=1.5,
                label="true signal")
        _a.scatter(X1d_tr[:, 0], y1d_tr, s=26, c="#B0B7C3", edgecolor="white",
                   linewidth=0.5, label="train data")
        _a.plot(_grid[:, 0], _Fgrid, c="#4C72B0", lw=2, label="$F_m(x)$")
        _a.set_title(f"m = {_m}")
        _a.text(0.03, 0.95,
                f"train RMSE {np.sqrt(((y1d_tr - _Ftr) ** 2).mean()):.3f}",
                transform=_a.transAxes, va="top", fontsize=9)

        _b = _axes[1, _j]
        _b.axhline(0, c="#999", lw=0.8)
        _b.scatter(X1d_tr[:, 0], y1d_tr - _Ftr, s=26, c="#DD8452",
                   edgecolor="none", label="residual $y - F_m$")
        if _m < len(gb1d.trees_):
            _b.plot(_grid[:, 0], gb1d.trees_[_m].predict(_grid), c="#55A868",
                    lw=2, label="next tree's fit")
        _b.set_ylim(-_rlim, _rlim)
        _b.set_xlabel("x")

    _axes[0, 0].set_ylabel("y")
    _axes[1, 0].set_ylabel("residual")
    _axes[0, 0].legend(loc="upper left", fontsize=8)
    _axes[1, 0].legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(X1d_te, X1d_tr, gb1d, np, plt, y1d_te, y1d_tr):
    # Learning Curve plot
    _Ftr = np.full(len(X1d_tr), gb1d.F0_)
    _Fte = np.full(len(X1d_te), gb1d.F0_)
    _tr, _te = [], []
    for _tree in gb1d.trees_:
        _Ftr += gb1d.learning_rate * _tree.predict(X1d_tr)
        _Fte += gb1d.learning_rate * _tree.predict(X1d_te)
        _tr.append(np.sqrt(((y1d_tr - _Ftr) ** 2).mean()))
        _te.append(np.sqrt(((y1d_te - _Fte) ** 2).mean()))

    _ms = np.arange(1, len(gb1d.trees_) + 1)
    _best = int(np.argmin(_te))

    _fig2, _ax2 = plt.subplots(figsize=(7, 4.5))
    _ax2.plot(_ms, _tr, c="#4C72B0", lw=2, label="train")
    _ax2.plot(_ms, _te, c="#DD8452", lw=2, label="test")
    _ax2.axvline(_best + 1, ls="--", c="#999", lw=1)
    _ax2.annotate(f"best test RMSE {_te[_best]:.3f} at m={_best + 1}",
                  xy=(_best + 1, _te[_best]), xytext=(0.45, 0.75),
                  textcoords="axes fraction", fontsize=9,
                  arrowprops=dict(arrowstyle="->", color="#666", lw=1))
    _ax2.set_xlabel("number of trees (m)")
    _ax2.set_ylabel("RMSE")
    _ax2.legend()
    _fig2.tight_layout()
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Decision Tree Implementation
    Identical to 'decision-trees.py'
    """)
    return


@app.cell(hide_code=True)
def _(Optional, dataclass, default_rng, np):
    @dataclass
    class Node:
        """
        A node in a decision tree.
        A node is either a leaf node which holds a value but no childen 
        or splits on (feature <= threshold) into left
        and right

        Attributes:
            feature: Index of feature used for split (none on leaves)
            threshold: Split point. X[feature] <= threshold go left. Else right.
            left: subtree for left side. None if a leaf node
            right: subtree for right side. None if a leaf node
            value: Prediction for samples at this leaf. None if not a leaf node.
            n_samples: No. of training samples that reached this node.
            gain: Impurity reduction from this node's split. 0.0 on leaf nodes
        """
        feature: Optional[int] = None
        threshold: Optional[float] = None
        left: Optional["Node"] = None
        right: Optional["Node"] = None
        value: Optional[float] = None
        n_samples: int = 0
        gain: float = 0.0

        @property
        def is_leaf(self) -> bool:
            return self.right is None and self.left is None

    def variance_reduction(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray):
        """Calculates the information gain for a split. Used for regression.
        """
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        if n_left == 0 or n_right == 0:
            return 0.0

        gain = np.var(y_parent) - (n_left/n * np.var(y_left) + n_right/n * np.var(y_right))

        return gain

    def entropy(y: np.ndarray):
        """Calculates Shannon entropy for a given split.
        """
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)

        return float(-np.sum(p * np.log2(p)))

    def entropy_reduction(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray):
        """Calculates the information gain for a split. Used for classification.
        """
        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        if n_left == 0 or n_right == 0:
            return 0.0

        gain = entropy(y_parent) - ((n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right))

        return gain

    class DecisionTree:
        """A tree, fit greedily by maximising a split criterion.
        Criterion is provided by subclasses (DecisionTreeRegressor & DecisionTreeClassifer)

        Attributes:
            max_depth: Maximum tree depth. 
            min_samples_split: A node with fewer samples becomes a leaf.
            max_features: Max no. of features used to fit a tree in Random Forest algorithm. None if fitting single decision tree
            criterion: Returns the gain for a split with higher bein1g better. 
                The specific type (variance or entropy) is provided by the subclasses.
        """

        def __init__(
            self,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: Optional[int] = None,
            random_state: Optional[int] = None,       
            criterion=variance_reduction,
        ):
            # Hyperparameters
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.max_features = max_features
            self.random_state= random_state
            self.criterion = criterion
            self.root_: Optional[Node] = None

        def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
            """Build the tree from training data.

            Args: 
                X: Feature matrix, shape (n_samples, n_features)
                y: Targets, shape (n_samples, )

            Returns:
                self
            """
            self.rng_  = default_rng(self.random_state)
            self.root_ = self._grow(X, y, depth=0)

            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict a value for each row of X.
            """
            yhat = []
            for row in X:
                prediction = self._predict_row(row)
                yhat.append(prediction)
            return np.array(yhat)

        def _grow(self, X: np.ndarray, y: np.ndarray, depth: int):
            """Recursively build the subtree for the rows (X, y)
            """

            n = len(y)

            stop = (
                # Stop when max depth is reached
                (self.max_depth is not None and depth >= self.max_depth)
                # Stop when too few rows to split
                or (n < self.min_samples_split)
                # Stop when y is pure
                or (len(np.unique(y)) == 1)
            )
            if stop:
                return Node(value=self._leaf_value(y), n_samples = n)

            feature, threshold, gain = self._best_split(X, y)

            # nothing beat gain of 0.0
            if feature is None:
                return Node(value=self._leaf_value(y), n_samples=n)

            mask = X[:, feature] <= threshold

            left = self._grow(X[mask], y[mask], depth + 1)
            right = self._grow(X[~mask], y[~mask], depth + 1)

            return Node(feature=feature, threshold=threshold, left=left, right=right, n_samples=n, gain=gain)

        def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
            """Search every (feature, threshold) pair; return the best.

            Returns:
                (feature_index, threshold, gain) or (None, None, 0.0) when no candidate split produces positive gain
            """
            best_gain = 0.0
            best_feature = None
            best_threshold = None

            n_features = X.shape[1]
            if self.max_features is None: 
                features = range(n_features)

            # chooses a random subset of size k features for random forest
            else:
                k = min(self.max_features, n_features)
                features = self.rng_.choice(n_features, size=k, replace=False)

            for feature in features:
                values = np.unique(X[:, feature]) 
                thresholds = (values[:-1] + values[1:]) / 2

                for threshold in thresholds:
                    mask = X[:, feature] <= threshold
                    # skips the threshold if it produces produces splits with leaf nodes with too little samples. 
                    if mask.sum() < self.min_samples_leaf or (~mask).sum() < self.min_samples_leaf:
                        continue

                    gain = self.criterion(y, y[mask], y[~mask])

                    if gain > best_gain:
                        best_gain = gain
                        best_threshold = threshold
                        best_feature = feature

            return best_feature, best_threshold, best_gain

        def _predict_row(self, x: np.ndarray) -> float:
            """Uses tree to give prediction for one row/sample.
            """
            node = self.root_

            while not node.is_leaf:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.value

    class DecisionTreeRegressor(DecisionTree):
        def __init__(self, criterion=variance_reduction, **kwargs):
            super().__init__(criterion=criterion, **kwargs)

        def _leaf_value(self, y: np.ndarray):
            """Return the prediction stored at leaf holding y. 
            For regression, this is the mean of all the samples at the leaf node.
            """
            return float(y.mean())

    class DecisionTreeClassifier(DecisionTree):
        def __init__(self, criterion=entropy_reduction, **kwargs):
            super().__init__(criterion=criterion, **kwargs)

        def _leaf_value(self, y: np.ndarray):
            """Return the prediction stored at leaf holding y. 
            For classification, this is the class with the majority number of samples at the leaf node.
            """
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

    return DecisionTreeRegressor, Node


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gradient Boosting Implementation
    """)
    return


@app.cell
def _(DecisionTreeRegressor, Node, Optional, np):
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Needed to convert model output to probabilities"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


    def leaf_for_row(node: Node, x: np.ndarray) -> Node:
        """Returns the node that the sample x's leaf belongs to.
        """
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    class GradientBoosting:
        """Base class holding the boosting loop.

        Builds an additive model F(x) = F0 + eta * sum_m h_m(x), where each h_m is a
        shallow regression tree fit to the negative gradient of the loss at the
        current scores. The loop is loss-agnostic; subclasses supply the pieces that
        depend on the loss:

            _initial_prediction   the constant F0 to start from        (required)
            _negative_gradient    what each tree is fit against        (required)
            predict               how a raw score becomes an answer    (required)
            _update_leaf_values   re-solve leaf values for this loss   (optional;
                                  a no-op by default, which is already correct
                                  for squared error)

        Attributes:
            n_estimators: Number of boosting rounds, i.e. trees in the ensemble.
            learning_rate: Shrinkage eta applied to every tree's contribution.
                Smaller values need more trees but usually generalise better.
            max_depth: Maximum depth of each tree. Boosting wants weak learners,
                so this stays deliberately small (3 is typical).
            min_samples_split: A node with fewer samples becomes a leaf.
            min_samples_leaf: Fewest samples a split may leave on either side.
            random_state: Currently unused -- stored, but never passed to the trees.
                Nothing in the algorithm is random: every tree searches every
                feature and there is no subsampling, so fits are deterministic.
            F0_: The constant the ensemble starts from. Set by fit.
            trees_: Fitted trees, in the order they were added. Set by fit.
        """

        def __init__(
            self,
            n_estimators: int = 100,
            learning_rate: float = 0.1,
            max_depth: Optional[int] = 3,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            random_state: Optional[int] = None,
        ):
            # Hyperparameters
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.random_state = random_state

            # Fitted state
            self.F0_: float = 0.0
            self.trees_: list[DecisionTreeRegressor] = []

        def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
            """Fit the ensemble by repeatedly fitting trees to the residual.
        
            Args:
                X: Feature matrix, shape (n_samples, n_features)
                y: Targets, shape (n_samples, )

            Returns:
                self
            """
            self.F0_ = self._initial_prediction(y)
            self.trees_ = []
            F = np.full(len(y), self.F0_, dtype=float)
        
            for m in range(self.n_estimators):
                r = self._negative_gradient(y, F)
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf
                ).fit(X, r)
            
                # Structure is fixed; the subclass may now re-solve each leaf's value.
                self._update_leaf_values(tree, X, y, F, r)

                F += self.learning_rate * tree.predict(X)
                self.trees_.append(tree)

            return self
        

        def _raw_predict(self, X: np.ndarray) -> np.ndarray:
            """The ensemble's raw prediction F: initial constant plus every tree's contribution.
            """
            F = np.full(len(X), self.F0_, dtype=float)
            for tree in self.trees_:
                f = tree.predict(X)
                F += self.learning_rate*f
            return F

        def _update_leaf_values(self, tree, X: np.ndarray, y: np.ndarray, F: np.ndarray, g: np.ndarray) -> None:
            """Optionally overwrite the tree's leaf values once its structure is fixed.
                For regression this should return none.
                The function is refined in the the class GradientBosstingClassifer
            """
            return None

    class GradientBoostingRegressor(GradientBoosting):
        def _initial_prediction(self, y: np.ndarray) -> float:
            """Inital prediction for regression is just the mean
            """
            f0 = float(y.mean())
            return f0

        def _negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
            """Negative gradient is the residual
            """
            r = y - F
            return r

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self._raw_predict(X)


    class GradientBoostingClassifier(GradientBoosting):
        """Binary classification. Expects y in {0, 1}."""

        def _initial_prediction(self, y: np.ndarray) -> float:
            p = y.mean()
            p = np.clip(p, 1e-6, 1-1e-6)
            return np.log(p/(1-p))

        def _negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
            """For classification the negative gradient is also the residual.
                F needs to converted to a probability first.
            """
            r = y - sigmoid(F)
            return r

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Probability of the positive class for each row of X."""
            proba = sigmoid(self._raw_predict(X))
            return proba

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predicted class label (0 or 1) for each row of X."""
            proba = self.predict_proba(X)
            return (proba >= 0.5).astype(int)

        def _update_leaf_values(self, tree, X: np.ndarray, y: np.ndarray, F: np.ndarray, r: np.ndarray) -> None:
            """Replace each leaf's mean-gradient with the Newton step for logistic loss.
            Formula is described in cell describing the algorithm        
            """

            p = sigmoid(F)
            h = p*(1-p)
            leaves = [leaf_for_row(tree.root_, x) for x in X]
            buckets = {}

            for i, leaf in enumerate(leaves):
                key = id(leaf)
                if key not in buckets:
                    buckets[key] = (leaf, [])
                buckets[key][1].append(i)

            for leaf, rows in buckets.values():
                leaf.value = r[rows].sum() / max(h[rows].sum(), 1e-12)

    return GradientBoostingClassifier, GradientBoostingRegressor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification
    """)
    return


@app.cell
def _(make_moons, plt):
    # Create binary classification dataset
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

    # Plot
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _colors = ["#4C72B0", "#DD8452"]
    for _k in range(2):
        _m = y == _k
        _ax.scatter(X[_m, 0], X[_m, 1], c=_colors[_k], label=f"class {_k}",
                    s=35, alpha=0.8, edgecolor="white", linewidth=0.5)
    _ax.set_xlabel("$x_1$")
    _ax.set_ylabel("$x_2$")
    _ax.set_title("make_moons (n=200, noise=0.2)")
    _ax.legend()
    _ax
    return X, y


@app.cell
def _(GradientBoostingClassifier, SkGB_clf, X, mo, train_test_split, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3).fit(Xtr, ytr)

    sk_clf = SkGB_clf(
        n_estimators      = 50,
        max_depth         = 3,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        learning_rate     = 0.1,
        random_state      = 0,
    ).fit(Xtr, ytr)

    tr_acc_mine = (clf.predict(Xtr) == ytr).mean()
    tr_acc_sk   = (sk_clf.predict(Xtr) == ytr).mean()

    te_acc_mine = (clf.predict(Xte) == yte).mean()
    te_acc_sk   = (sk_clf.predict(Xte) == yte).mean()

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
    ## Regression
    """)
    return


@app.cell
def _(load_diabetes):
    # Loads the diabetes dataset
    diabetes = load_diabetes(scaled=False)
    X_reg, y_reg, names_reg = diabetes.data, diabetes.target, list(diabetes.feature_names)
    return X_reg, y_reg


@app.cell
def _(
    GradientBoostingRegressor,
    SkGB_reg,
    X_reg,
    mo,
    np,
    train_test_split,
    y_reg,
):
    Xtr_reg, Xte_reg, ytr_reg, yte_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=0)

    reg = GradientBoostingRegressor(n_estimators=10, max_depth=3).fit(Xtr_reg, ytr_reg)

    sk_reg = SkGB_reg(
        n_estimators      = 10,
        max_depth         = 3,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        learning_rate     = 0.1,
        random_state      = 0,
    ).fit(Xtr_reg, ytr_reg)

    def rmse(yhat, y):
        """Computes RMSE for comparison regression trees
        """
        return np.sqrt(np.mean((yhat - y)**2))

    tr_rmse_mine = rmse(reg.predict(Xtr_reg), ytr_reg)
    te_rmse_mine = rmse(reg.predict(Xte_reg), yte_reg)

    tr_rmse_sk = rmse(sk_reg.predict(Xtr_reg), ytr_reg)
    te_rmse_sk = rmse(sk_reg.predict(Xte_reg), yte_reg)

    mo.md(
        f"""
        **from scratch**
        ----------------
        Training Accuracy: {tr_rmse_mine:.4f} \n
        Test Accuracy:     {te_rmse_mine:.4f}

        **sklearn**
        ----------------
        Training Accuracy: {tr_rmse_sk:.4f} \n
        Test Accuracy:     {te_rmse_sk:.4f} \n

        RMSE match between sklearn and from scratch implementation.
        """
    )
    return


if __name__ == "__main__":
    app.run()
