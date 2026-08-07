import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from dataclasses import dataclass
    from typing import Optional
    import numpy as np

    return Optional, dataclass, np


@app.cell
def _(Optional, dataclass):
    @dataclass
    class Node:
        feature: Optional[int] = None
        threshold: Optional[float] = None
        left: Optional["Node"] = None
        right: Optional["Node"] = None
        value: Optional[float] = None
        n_samples: int = 0
        gain: float = 0.0

        @property
        def is_leaf(self) -> bool:
            return self.value is not None


    return (Node,)


@app.cell
def _(np):
    def variance_reduction(y_parent: np.ndarray,
                          y_left: np.ndarray,
                          y_right: np.ndarray):


        n = len(y_parent)
        n_left = len(y_left)
        n_right = len(y_right)

        if n_left == 0 or n_right == 0:
            return 0.0
    
        gain = np.var(y_parent) - (n_left/n * np.var(y_left) + n_right/n * np.var(y_right))

        return gain



    return (variance_reduction,)


@app.cell
def _(Node, Optional, np, variance_reduction):
    class DecisionTree:
        """A regression tree, fit greedily by maximising variance reduction."""

        def __init__(
            self,
            max_depth: Optional[int] = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            criterion=variance_reduction,
        ):
            # Hyperparameters: everything describing *how* to fit.
            # Nothing learned from data belongs in __init__.
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.criterion = criterion

            # Learned state. The trailing underscore is the sklearn convention
            # for "this gets filled in by fit()".
            self.root_: Optional[Node] = None

        # ------------------------------------------------ public API ---

        def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
            """Build the tree from training data.

            TODO
              1. self.root_ = self._grow(X, y, depth=0)
              2. return self   -- lets you chain: tree.fit(X, y).predict(X)
            """
            ...

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict a value for each row of X.

            TODO
              Send each row down the tree and collect the leaf values:
              np.array([self._predict_row(row) for row in X])
            """
            ...

        # ------------------------------------------------- internals ---

        def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
            """Recursively build a subtree for (X, y); return its root Node.

            Three parts:

            1. STOP? Return a leaf if any of these hold:
                 - self.max_depth is not None and depth >= self.max_depth
                 - len(y) < self.min_samples_split
                 - y is already pure: np.all(y == y[0])
               A leaf is Node(value=y.mean(), n_samples=len(y)).
               (value= is what makes Node.is_leaf True -- look at your Node.)

            2. SPLIT: feature, threshold, gain = self._best_split(X, y)
               If feature is None, no split beat doing nothing -- return a leaf.

            3. RECURSE: build the boolean mask, then
                 left  = self._grow(X[mask],  y[mask],  depth + 1)
                 right = self._grow(X[~mask], y[~mask], depth + 1)
               and return the internal node:
                 Node(feature=..., threshold=..., left=left, right=right,
                      n_samples=len(y), gain=gain)
            """
            ...

        def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
            """Search every (feature, threshold) pair; return the best.

            Returns (feature_index, threshold, gain), or (None, None, 0.0)
            when nothing improves on a leaf.

            Sketch (pseudocode, not runnable):

                best_gain, best_feature, best_threshold = 0.0, None, None

                for feature in range(X.shape[1]):        # each column of X
                    values = np.unique(X[:, feature])    # sorted + deduped
                    # candidate thresholds = midpoints between adjacent values
                    for threshold in (values[:-1] + values[1:]) / 2:
                        mask = X[:, feature] <= threshold
                        # respect min_samples_leaf: skip if either side too small
                        gain = self.criterion(y, y[mask], y[~mask])
                        # keep it if gain > best_gain

                return best_feature, best_threshold, best_gain

            Note your criterion already returns 0.0 for an empty side, and
            best_gain starts at 0.0, so degenerate splits can never win.
            """
            best_gain = 0.0
            best_feature = None
            best_threshold = None

            for feature in range(X.shape[1]):
                values = np.unique(X[:, feature]) 

                thresholds = (values[:-1] + values[1:]) / 2
                for threshold in thresholds:
                    mask = X[:, feature] <= threshold
                    gain = self.criterion(y, y[mask], y[~mask])
                
                    if gain > best_gain:
                        best_gain = gain
                        best_threshold = threshold
                        best_feature = feature
                    
            return best_feature, best_threshold, best_gain

        def _predict_row(self, x: np.ndarray) -> float:
            """Walk one sample from the root down to a leaf; return its value.

            TODO
              node = self.root_
              while not node.is_leaf:
                  go left if x[node.feature] <= node.threshold, else right
              return node.value
            """
            ...


    return


if __name__ == "__main__":
    app.run()
