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

    from numpy.random import SeedSequence, default_rng

    return Optional, SeedSequence, dataclass, default_rng, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest
    """)
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Decision Tree Implementation
    See 'decision-trees.py' for implementation details and comparison
    """)
    return


@app.cell
def _(Optional, dataclass, np):
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


@app.cell
def tree_checks(DecisionTreeRegressor, mo, np):
    def _tree_checks():
        """Three trees that should all differ from each other. Do they?"""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        y = 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * rng.normal(size=200)

        def fingerprint(tree):
            """Flatten a fitted tree into a comparable list of its splits."""
            out = []

            def walk(node):
                if node.is_leaf:
                    out.append(("leaf", round(float(node.value), 6)))
                    return
                out.append((node.feature, round(float(node.threshold), 6)))
                walk(node.left)
                walk(node.right)

            walk(tree.root_)
            return out

        def fit(**kwargs):
            return fingerprint(DecisionTreeRegressor(max_depth=3, **kwargs).fit(X, y))

        seed_a = fit(max_features=1, random_state=0)
        seed_b = fit(max_features=1, random_state=999)
        full_search = fit(max_features=None, random_state=0)

        checks = [
            ("two different seeds build two different trees", seed_a != seed_b),
            ("max_features=1 differs from the full greedy search", seed_a != full_search),
        ]

        lines = ["**Is `max_features` actually restricting the split search?**", ""]
        lines += ["| | check |", "|---|---|"]
        lines += [f"| {'PASS' if ok else 'FAIL'} | {label} |" for label, ok in checks]
        lines += [""]
        if all(ok for _, ok in checks):
            lines += ["`max_features` works. Your trees can be decorrelated."]
        else:
            lines += [
                "`max_features` is being ignored: every tree is the same greedy tree,",
                "so a forest built on it would only be bagging. Look at the loop header",
                "in `DecisionTree._best_split` -- what is it iterating over, and what did",
                "you just spend four lines computing?",
            ]
        return mo.md("\n".join(lines))


    _tree_checks()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Random Forest Implementation
    Based on decision tree implementation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    #### Five questions to answer before writing the code

    **1. Bootstrap.** Each tree trains on `n` rows drawn *with replacement* from the `n`
    training rows. Roughly what fraction of the original rows does any given tree never
    see? It converges to a constant, and that constant is what makes out-of-bag scoring
    possible.

    **2. Seeding.** `DecisionTree.fit` builds `self.rng_` from `self.random_state`. If the
    forest passes the same `random_state` to all `n_estimators` trees, what do you get?
    How do you give each tree an independent stream that is still reproducible from one
    forest-level seed?

    **3. `max_features` default.** `None` means "search every feature" -- the wrong default
    for a forest. The convention is `sqrt(n_features)` for classification and
    `n_features // 3` for regression. Why would classification want *fewer* candidate
    features per split than regression?

    **4. Aggregation.** Regression averages the tree predictions. Classification cannot
    average labels. Your `DecisionTreeClassifier._leaf_value` returns a hard label, not a
    class distribution -- what does that force your forest's `predict` to do, and what
    capability do you give up by storing only the argmax at each leaf?

    **5. Shared structure.** `RandomForestRegressor` and `RandomForestClassifier` differ in
    exactly three places: which tree class to build, the default `max_features`, and how to
    combine predictions. You already solved this shape once with `DecisionTree` and its two
    subclasses -- mirror it.
    """)
    return


@app.cell
def random_forest(
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Optional,
    SeedSequence,
    default_rng,
    np,
):
    class RandomForest:
        """An ensemble of decision trees, each fit on a bootstrap sample of the rows
        and restricted to a random subset of features at every split.

        Subclasses supply the three varying pieces:
            tree_cls: the DecisionTree subclass to build
            _default_max_features(n_features): used when max_features is None
            _aggregate(tree_preds): combines a (n_estimators, n_samples) array into
                a single (n_samples,) prediction
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
            """Build the ensemble.

            TODO:
              1. Resolve the effective max_features: self.max_features if given,
                 otherwise self._default_max_features(X.shape[1]).
              2. Produce n_estimators independent seeds from self.random_state.
                 (np.random.SeedSequence(self.random_state).spawn(...) is one way.)
              3. For each tree: draw a bootstrap sample, construct self.tree_cls(...)
                 with the resolved max_features and that tree's own seed, fit it,
                 and append it to self.trees_.

            Returns:
                self
            """
            k = k = self.max_features if self.max_features is not None else X.shape[1]
            self.trees_ = []
            # Generate seeds for fitting each tree
            ss = SeedSequence(self.random_state)
            children = ss.spawn(self.n_estimators)
            seeds = [default_rng(c) for c in children]

            for seed in seeds:
                rng = default_rng(seed)

                if self.bootstrap:
                    Xs, ys = self._bootstrap(X, y, rng)
                else:
                    Xs, ys = X, y

                tree = self.tree_cls(
                    max_depth         = self.max_depth,
                    min_samples_split = self.min_samples_split,
                    min_samples_leaf  = self.min_samples_leaf,
                    max_features      = k,
                    random_state      = seed,
                )
                tree.fit(Xs, ys)
                self.trees_.append(tree)

            return self

        def _bootstrap(self, X: np.ndarray, y: np.ndarray, rng) -> tuple:
            """Creates new dataset by randomly drawing samples from original dataset
               randomly with replacement.
            """
            n = X.shape[0]
            idx = rng.integers(0, n, size= n)
        
            return X[idx], y[idx]
                        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict by combining every tree's prediction.

            TODO: build a (n_estimators, n_samples) array of per-tree predictions,
            then hand it to self._aggregate.
            """
            raise NotImplementedError


    class RandomForestRegressor(RandomForest):
        tree_cls = DecisionTreeRegressor

        def _default_max_features(self, n_features: int) -> int:
            """TODO: the regression convention. Never return less than 1."""
            raise NotImplementedError

        def _aggregate(self, tree_preds: np.ndarray) -> np.ndarray:
            """TODO: combine (n_estimators, n_samples) -> (n_samples,)."""
            raise NotImplementedError


    class RandomForestClassifier(RandomForest):
        tree_cls = DecisionTreeClassifier

        def _default_max_features(self, n_features: int) -> int:
            """TODO: the classification convention. Never return less than 1."""
            raise NotImplementedError

        def _aggregate(self, tree_preds: np.ndarray) -> np.ndarray:
            """TODO: a vote, not a mean. np.unique(..., return_counts=True) per column
            works, though there are tidier ways."""
            raise NotImplementedError

    return


if __name__ == "__main__":
    app.run()
