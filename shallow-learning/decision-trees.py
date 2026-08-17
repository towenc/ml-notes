import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_diabetes, load_iris
    from sklearn.tree import DecisionTreeRegressor as SKRtree, DecisionTreeClassifier as SKCtree

    return (
        Optional,
        SKCtree,
        SKRtree,
        dataclass,
        load_diabetes,
        load_iris,
        mo,
        np,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Decision Trees
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Trees are "grown" according to the following algorithm:
    1. All samples start at the root node.
    2. Calculate information gain (entropy reduction for classification or variance reduction for regression) for all possible features, taking the midpoint between samples as the thresholds
    3. Pick the feature and threshold that gives the maximum information gain.
    5. Split the dataset according to the selected feature and threshold, creating left and right branches of the tree where the left branch holds the values that are less than or equal to the threshold.
    6. Keep repeating the splitting process until stopping criteria is met
        - stop when max depth is reached (max depth)
        - stop when there are too few rows to split (min samples)
        - stop when purity at a leaf node is 1.

    **Information gain** is defined as:
    $$\text{Information gain} = I^{\text{parent}} - \left( w^{\text{left}} I^{\text{left}} + w^{\text{right}} I^{\text{right}} \right)$$
    Where $I$ is the impurity measure and $w$ is the ratio of samples in the branch ($w^{left} = \frac{n^{left}}{n^{parent}}$)

    $I$ changes depending on the task. For regression, $I$ is the variance:

    Let $\bar{y} = \frac{1}{m} \sum_{i=1}^{m} y^{(i)}$ at the node:
    $$\text{Var} = \frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - \bar{y} \right)^2$$

    For classification, in this example the entropy $H$ is used.

    With $K$ classes, let $p_k$ be the fraction of examples at the node with
    $y^{(i)} = k$:

    $$p_k = \frac{1}{m} \sum_{i=1}^{m} \mathbb{1}\left[ y^{(i)} = k \right]$$

    $$H = -\sum_{k=1}^{K} p_k \log_2(p_k)$$
    """)
    return


@app.cell
def _(Optional, dataclass):
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

    return (Node,)


@app.cell
def _(np):
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

    return entropy_reduction, variance_reduction


@app.cell
def _(Node, Optional, entropy_reduction, np, variance_reduction):
    class DecisionTree:
        """A tree, fit greedily by maximising a split criterion.
        Criterion is provided by subclasses (DecisionTreeRegressor & DecisionTreeClassifer)

        Attributes:
            max_depth: Maximum tree depth. 
            min_samples_split: A node with fewer samples becomes a leaf.
            criterion: Returns the gain for a split with higher being better. The specific type (variance or entropy) is provided by the subclasses.
        """

        def __init__(
            self,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            criterion=variance_reduction,
        ):
            # Hyperparameters
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
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
            ...
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

            for feature in range(X.shape[1]):
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

    return DecisionTreeClassifier, DecisionTreeRegressor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ###Helper Functions
    """)
    return


@app.cell
def _(Node, mo, np):
    def tree_lines(node, names, prefix="", conn=""):
        """Render a tree as a list of box-drawing lines, one per node."""
        if node.is_leaf:
            label = f"predict {node.value:.2f} (n={node.n_samples})"
        else:
            label = f"{names[node.feature]} <= {node.threshold:.2f} (n={node.n_samples})"

        lines = [prefix + conn + label]

        if not node.is_leaf:
            # A node's descendants keep a trailing vertical bar only while more
            # siblings are still to come; the last child's subtree gets blank space.
            kid = prefix + ("    " if conn == "└── " else "│   " if conn == "├── " else "")
            lines += tree_lines(node.left, names, kid, "├── ")
            lines += tree_lines(node.right, names, kid, "└── ")

        return lines


    def render_tree(node, names):
        return mo.plain_text("\n".join(tree_lines(node, names)))

    def from_sklearn(est, i=0):
        """Adapt a fitted sklearn tree into our Node type to render it the same way.
        """
        t = est.tree_
        left, right = t.children_left[i], t.children_right[i]

        if left == -1:
            raw = t.value[i][0]
            # classifiers store class proportions; regressors a single mean
            value = est.classes_[raw.argmax()] if hasattr(est, "classes_") else float(raw[0])
            return Node(value=value, n_samples=int(t.n_node_samples[i]))

        return Node(
            feature=int(t.feature[i]),
            threshold=float(t.threshold[i]),
            left=from_sklearn(est, left),
            right=from_sklearn(est, right),
            n_samples=int(t.n_node_samples[i]),
        )

    def rmse(yhat, y):
        """Computes RMSE for comparison regression trees
        """
        return np.sqrt(np.mean((yhat - y)**2))

    return from_sklearn, render_tree, rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Classification
    """)
    return


@app.cell
def _(load_iris, plt):
    # Load the iris dataset.
    iris = load_iris()
    X, y, names, classes = iris.data, iris.target, list(iris.feature_names), iris.target_names

    # Plot
    fig, ax = plt.subplots(figsize=(6,5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for k in range(3):
        m = y == k
        ax.scatter(X[m, 2], X[m, 3], c=colors[k], label=classes[k],
                   s=35, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(names[2]); ax.set_ylabel(names[3])
    ax.legend()
    return X, names, y


@app.cell
def _(
    DecisionTreeClassifier,
    SKCtree,
    X,
    from_sklearn,
    mo,
    names,
    render_tree,
    y,
):
    _depth = 3

    clf = DecisionTreeClassifier(max_depth=_depth).fit(X, y)
    sk_clf = SKCtree(criterion="entropy", max_depth=_depth, random_state=42).fit(X, y)

    _acc_mine = (clf.predict(X) == y).mean()
    _acc_sk = (sk_clf.predict(X) == y).mean()

    mo.hstack(
        [
            mo.vstack([
                mo.md(f"**from scratch** | accuracy `{_acc_mine:.4f}`"),
                render_tree(clf.root_, names),
            ]),
            mo.vstack([
                mo.md(f"**sklearn** | accuracy `{_acc_sk:.4f}`"),
                render_tree(from_sklearn(sk_clf), names),
            ]),
        ],
        widths="equal",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fitted trees from sklearn and from scratch implementation are exactly the same.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Regression
    """)
    return


@app.cell
def _(load_diabetes):
    # Loads the diabetes dataset
    diabetes = load_diabetes(scaled=False)
    X_reg, y_reg, names_reg = diabetes.data, diabetes.target, list(diabetes.feature_names)
    return X_reg, names_reg, y_reg


@app.cell
def _(
    DecisionTreeRegressor,
    SKRtree,
    X_reg,
    from_sklearn,
    mo,
    names_reg,
    render_tree,
    rmse,
    y_reg,
):
    _depth = 3

    reg = DecisionTreeRegressor(max_depth=_depth).fit(X_reg, y_reg)
    sk_reg = SKRtree(criterion="squared_error", max_depth=_depth, random_state=42).fit(X_reg, y_reg)

    rmse_reg = rmse(reg.predict(X_reg), y_reg)
    rmse_sk_reg = rmse(sk_reg.predict(X_reg), y_reg)

    mo.hstack(
        [
            mo.vstack([
                mo.md(f"**from scratch** | **RMSE:** {rmse_reg:.2f}"),
                render_tree(reg.root_, names_reg),
            ]),
            mo.vstack([
                mo.md(f"**sklearn** | **RMSE:** {rmse_sk_reg:.2f}"),
                render_tree(from_sklearn(sk_reg), names_reg),
            ]),
        ],
        widths="equal",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Fitted trees from sklearn and from scratch are exactly the same.
    """)
    return


if __name__ == "__main__":
    app.run()
