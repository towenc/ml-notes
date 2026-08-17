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
    from sklearn.datasets import load_diabetes, load_iris
    from sklearn.ensemble import RandomForestClassifier as SkRF_clf, RandomForestRegressor as SkRF_reg
    from sklearn.model_selection import train_test_split

    return (
        Optional,
        SkRF_clf,
        SkRF_reg,
        dataclass,
        default_rng,
        load_diabetes,
        load_iris,
        mo,
        np,
        plt,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Random Forest
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The random forest algorithm builds on the decision tree algorithm. It ensembles many decision trees fit on a random subset of features on bootstrapped data.

    The algorithm is as follows:

    For every tree you will fit:
    1. Sample the data with replacement to create a new dataset of the same size.
    2. Choose a random subset of the features.
    3. Fit a single decision tree.

    Aggregate the predictions of every tree you fit as the final prediction.
    For regression, this is taking the mean of the predictions. For classification, you take the majority vote of all the decision trees.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Decision Tree Implementation
    See 'decision-trees.py' for implementation details and comparison
    """)
    return


@app.cell
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

    return DecisionTreeClassifier, DecisionTreeRegressor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Random Forest Implementation
    Based on decision tree implementation
    """)
    return


@app.cell
def random_forest(
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Optional,
    default_rng,
    np,
):
    class RandomForest:
        """
        An ensemble of decision trees, each fit on a bootstrap sample of the rows
        and restricted to a random subset of features at every split.

        Subclasses [RandomForestRegressor, RandomForestClassifier] define:
            tree_cls: the DecisionTree subclass to build (regression or classification)
            _default_max_features(n_features): used when max_features is None
            _aggregate(tree_preds): combines a (n_estimators, n_samples) array into
                a single (n_samples,) prediction

        Attributes:]
            trees_: the fitted estimators, given by fit()    
        """
        tree_cls = None

        def __init__(
            self,
            n_estimators: int = 100,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: Optional[int] = None,
            bootstrap: bool = True,
            random_state: Optional[int] = None,
        ):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.max_features = max_features
            self.bootstrap = bootstrap
            self.random_state = random_state

            self.trees_: list = []

        def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
            """
            Builds the ensemble. Fits n_estimators trees, each on a boostrapped dataset when self.bootstrap

            Returns:
                self
            """

        
            if self.max_features is not None:
                k = self.max_features
            else:
                k = self._default_max_features(X.shape[1])
   
            self.trees_ = []
    
            # Generate seeds for bootstrapping and fitting each tree
            rng = default_rng(self.random_state)
            seeds = rng.integers(0, 2**32, size=(self.n_estimators, 2))

            for boot_seed, tree_seed in seeds:

                if self.bootstrap:
                    Xs, ys = self._bootstrap(X, y, default_rng(boot_seed))
                else:
                    Xs, ys = X, y

                tree = self.tree_cls(
                    max_depth         = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    min_samples_leaf  = self.min_samples_leaf,
                    max_features      = k,
                    random_state      = int(tree_seed),
                )
                tree.fit(Xs, ys)
                self.trees_.append(tree)

            return self

        def _bootstrap(self, X: np.ndarray, y: np.ndarray, rng) -> tuple:
            """
            Creates new dataset by randomly drawing samples from original dataset
            with replacement

            Returns:
                (X_sample, y_sample), both length n
            """
            n = X.shape[0]
            idx = rng.integers(0, n, size= n)

            return X[idx], y[idx]
                
        def predict(self, X: np.ndarray) -> np.ndarray:
            """
            Predict by combining every tree's prediction via _aggregate
            Majority vote for classification, mean for regression.

            Returns:
                (n_samples,) predictions
            """
            predictions = []
            for tree in self.trees_:
                predictions.append(tree.predict(X))
    
            return self._aggregate(np.array(predictions))


    class RandomForestRegressor(RandomForest):
        tree_cls = DecisionTreeRegressor

        def _default_max_features(self, n_features: int) -> int:
            """Returns the default number of features to use when fitting a single tree.
            Minimum number of features will be 1.        
            """
            max_features = max(1, n_features // 3)
            return max_features

        def _aggregate(self, tree_preds: np.ndarray) -> np.ndarray:
            """Aggregates the predictions of all the trees using the mean."""

            yhat = np.mean(tree_preds, axis=0)
            return yhat


    class RandomForestClassifier(RandomForest):
        tree_cls = DecisionTreeClassifier

        def _default_max_features(self, n_features: int) -> int:
            """Returns the default number of features to use when fitting a single tree.
            Minimum number of features will be 1
            """
            max_features = max(1, int(np.sqrt(n_features)))
            return max_features

        def _aggregate(self, tree_preds: np.ndarray) -> np.ndarray:
            """Aggregates the predictions of all the trees using the majority vote"""

            yhat = []
            for j in range(tree_preds.shape[1]):
                classes, counts = np.unique(tree_preds[:, j], return_counts=True)
                yhat.append(classes[counts.argmax()])
    
            return np.array(yhat)

    return RandomForestClassifier, RandomForestRegressor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison and Evaluation
    """)
    return


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
    return X, y


@app.cell
def _(RandomForestClassifier, SkRF_clf, X, mo, train_test_split, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = RandomForestClassifier(n_estimators=3, max_depth=3).fit(Xtr, ytr)

    sk_clf = SkRF_clf(
        n_estimators      = 3,
        max_depth         = 3,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        max_features      = "sqrt",
        bootstrap         = True,
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

        Accuraries match between sklearn and from scratch implementation. They cannot exactly match due to randomness
        """
    )
    return Xte, Xtr, yte, ytr


@app.cell
def _(RandomForestClassifier, Xte, Xtr, mo, plt, yte, ytr):
    n_trees = [1, 2, 5, 10, 20, 50, 100, 200]
    tr_acc = []
    te_acc = []

    for n in n_trees:
        mine = RandomForestClassifier(n_estimators=n, max_depth=3).fit(Xtr, ytr)
        tr_acc.append((mine.predict(Xtr) == ytr).mean())
        te_acc.append((mine.predict(Xte) == yte).mean())

    plt.plot(n_trees, tr_acc, "o--", label="Training Accuracy")
    plt.plot(n_trees, te_acc, "s--", label="Test Accuracy")
    plt.xscale("log")
    plt.xlabel("n_estimators")
    plt.ylabel("accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    mo.md(
        """
        Learning curve is also as expected. As the number of trees fit (n_estimators) increases after 1 the accuracies increase.
        """
    )
    return (n_trees,)


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
    return X_reg, y_reg


@app.cell
def _(RandomForestRegressor, SkRF_reg, X_reg, mo, np, train_test_split, y_reg):
    Xtr_reg, Xte_reg, ytr_reg, yte_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=0)

    reg = RandomForestRegressor(n_estimators=100, max_depth=15).fit(Xtr_reg, ytr_reg)

    k_reg = max(1, X_reg.shape[1] // 3)
    sk_reg = SkRF_reg(
        n_estimators      = 100,
        max_depth         = 15,
        min_samples_split = 2,
        min_samples_leaf  = 1,
        max_features      = k_reg,
        bootstrap         = True,
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

        RMSE match between sklearn and from scratch implementation. They cannot exactly match due to randomness
        """
    )
    return Xte_reg, Xtr_reg, rmse, yte_reg, ytr_reg


@app.cell
def _(
    RandomForestRegressor,
    Xte_reg,
    Xtr_reg,
    mo,
    n_trees,
    plt,
    rmse,
    yte_reg,
    ytr_reg,
):
    tr_rmse = []
    te_rmse = []

    for n_reg in n_trees:
        mine_reg = RandomForestRegressor(n_estimators=n_reg, max_depth=3).fit(Xtr_reg, ytr_reg)
        tr_rmse.append(rmse(mine_reg.predict(Xtr_reg), ytr_reg))
        te_rmse.append(rmse(mine_reg.predict(Xte_reg), yte_reg))

    plt.plot(n_trees, tr_rmse, "o--", label="Training RMSE")
    plt.plot(n_trees, te_rmse, "s--", label="Test RMSE")
    plt.xscale("log")
    plt.xlabel("n_estimators")
    plt.ylabel("RMSE")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    mo.md(
        """
        Learning curve is also as expected. As the number of trees fit (n_estimators) increases after 1 the RMSE decreases.
        """
    )
    return


if __name__ == "__main__":
    app.run()
