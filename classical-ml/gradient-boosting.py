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


    return Optional, dataclass, default_rng, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gradient Boosting
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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
    3. Predict $\sigma(F_M(x))$, thresholding at 0.5 for a hard label.
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
    Based on the decision tree implementation.

    With squared error loss $L = \frac{1}{2}\sum_i (y_i - F(x_i))^2$, the derivative with respect to a single prediction is

    $$\frac{\partial L}{\partial F(x_i)} = -(y_i - F(x_i)) = -r_i$$

    So the residual **is** the negative gradient. "Fit a tree to the residuals" and "take a gradient descent step, where the step direction is whatever a tree can represent" are the same instruction. That is the whole reason the method generalises: swap the loss, recompute $-\partial L / \partial F$, and the loop is unchanged.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Boosting a different loss

    For binary classification the ensemble does **not** predict a class, or even a probability. It predicts a raw score $F(x)$ interpreted as **log-odds**, squashed into a probability only at the end:

    $$p(x) = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$$

    Under the logistic loss $L = -\sum_i \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]$, the derivative with respect to the raw score collapses to something remarkably clean:

    $$\frac{\partial L}{\partial F(x_i)} = p_i - y_i \qquad \Longrightarrow \qquad -\frac{\partial L}{\partial F(x_i)} = y_i - \sigma(F(x_i))$$

    Compare the two losses side by side:

    | | regression (squared error) | binary classification (logistic) |
    |---|---|---|
    | what $F$ means | the prediction itself | log-odds |
    | $F_0$ | $\text{mean}(y)$ | $\log \frac{\bar{y}}{1 - \bar{y}}$ |
    | negative gradient | $y - F$ | $y - \sigma(F)$ |
    | final output | $F$ | $\sigma(F) \geq 0.5$ |

    Only three rows differ, and the loop that consumes them is identical. That is what the base class below captures: `fit` holds the boosting loop, and each subclass supplies its initial value, its gradient, and how to turn a raw score into an answer.
    """)
    return


@app.cell
def _(DecisionTreeRegressor, Node, Optional, np):
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Squash raw log-odds scores into probabilities, clipped for numerical safety."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


    def leaf_for_row(node: Node, x: np.ndarray) -> Node:
        """Walk the tree to the leaf that x lands in, returning the Node itself.

        DecisionTree._predict_row returns node.value, which is enough to predict but
        not enough to *modify* a leaf. Lives here rather than on DecisionTree so the
        transplanted tree cell stays byte-identical to 'decision-trees.py'.
        """
        # TODO walk left/right on (x[node.feature] <= node.threshold) until node.is_leaf,
        #      then return the node
        raise NotImplementedError


    class GradientBoosting:
        """Base class holding the boosting loop.

        Subclasses supply the three loss-specific pieces: where the ensemble starts,
        what the negative gradient is, and how a raw score becomes an answer.
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
            """Fit the ensemble by repeatedly fitting trees to the negative gradient.

            Args:
                X: Feature matrix, shape (n_samples, n_features)
                y: Targets, shape (n_samples, )

            Returns:
                self
            """
            # TODO 1. self.F0_ = self._initial_prediction(y)
            # TODO 2. F = running raw score for the training rows, filled with self.F0_
            # TODO 3. self.trees_ = []   (so refitting does not extend an old ensemble)
            # TODO 4. Loop self.n_estimators times:
            #           a. g = self._negative_gradient(y, F)
            #           b. fit a DecisionTreeRegressor on (X, g), passing max_depth /
            #              min_samples_split / min_samples_leaf through
            #           c. F = F + self.learning_rate * tree.predict(X)
            #           d. self.trees_.append(tree)
            # TODO 5. return self

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
            """The ensemble's raw score F(X): initial constant plus every tree's contribution.

            This is the prediction for regression and the log-odds for classification,
            so it lives on the base class and each subclass decides how to interpret it.
            """
            # TODO start from an array of shape (len(X),) filled with self.F0_,
            #      then add self.learning_rate * tree.predict(X) for every fitted tree

            yhat = np.full(len(X), self.F0_, dtype=float)
            for tree in self.trees_:
                f = tree.predict(X)
                yhat += self.learning_rate*f
            return yhat

        def _update_leaf_values(self, tree, X: np.ndarray, y: np.ndarray, F: np.ndarray, g: np.ndarray) -> None:
            """Optionally overwrite the tree's leaf values once its structure is fixed.

            The tree stores the mean of the gradients falling in each leaf. That is a
            descent direction, but not the best step size for this loss. A subclass may
            replace each leaf value with the Newton step

                gamma = sum(g_i) / sum(h_i)     over the rows landing in that leaf

            where h_i is the second derivative of the loss at row i.

            A no-op here, and correctly so for squared error: there h_i = 1, so gamma is
            the mean of the residuals -- exactly what the leaf already holds.
            """
            return None

        def _initial_prediction(self, y: np.ndarray) -> float:
            """The constant F_0 that minimises this loss over y, before any tree is fit."""
            raise NotImplementedError

        def _negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
            """-dL/dF evaluated at the current raw scores. This is what each tree is fit against."""
            raise NotImplementedError

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Turn raw scores into the answer this estimator is supposed to return."""
            raise NotImplementedError


    class GradientBoostingRegressor(GradientBoosting):
        def _initial_prediction(self, y: np.ndarray) -> float:
            # TODO the constant minimising squared error over y
            f0 = float(y.mean())
            return f0

        def _negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
            # TODO the residual]
            r = y - F
            return r

        def predict(self, X: np.ndarray) -> np.ndarray:
            # TODO for regression the raw score already is the prediction
            return self._raw_predict(X)


    class GradientBoostingClassifier(GradientBoosting):
        """Binary classification. Expects y in {0, 1}."""

        def _initial_prediction(self, y: np.ndarray) -> float:
            # TODO the log-odds of the base rate, log(p / (1 - p)) where p = y.mean().
            #      Clip p away from exactly 0 and 1 or a single-class y gives +/- inf.
            p = y.mean()
            p = np.clip(p, 1e-6, 1-1e-6)
            return np.log(p/(1-p))

        def _negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
            # TODO y - sigmoid(F).  Note the trees are still fit on this continuous
            #      quantity, never on the labels themselves.
            r = y - sigmoid(F)
            return r

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Probability of the positive class for each row of X."""
            # TODO squash the raw score
            proba = sigmoid(self._raw_predict(X))
            return proba

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predicted class label (0 or 1) for each row of X."""
            # TODO threshold the probability at 0.5
            proba = self.predict_proba(X)
            return (proba >= 0.5).astype(int)

        def _update_leaf_values(self, tree, X: np.ndarray, y: np.ndarray, F: np.ndarray, g: np.ndarray) -> None:
            """Replace each leaf's mean-gradient with the Newton step for logistic loss."""
            # TODO 1. p = sigmoid(F), then h = p * (1 - p)  -- the per-row Hessian
            # TODO 2. group training rows by the leaf they land in:
            #           leaves = [leaf_for_row(tree.root_, x) for x in X]
            #         then bucket row indices by id(leaf), since Node is not hashable by value
            # TODO 3. for each leaf, set  leaf.value = g[rows].sum() / h[rows].sum()
            #         Guard the denominator, e.g. max(h[rows].sum(), 1e-12): a leaf whose
            #         rows are all confidently classified has h -> 0 and would blow up.
            #
            # Mutating leaf.value in place is all that is needed -- fit() calls
            # tree.predict(X) on the very next line and picks up the new values.
            raise NotImplementedError


    return


if __name__ == "__main__":
    app.run()
