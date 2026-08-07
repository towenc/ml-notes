import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


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


    return


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

    

    return


app._unparsable_cell(
    r"""
    class DecisionTree:
        def __init__(self,
                    max_depth: Optional[int] = 0,
                    min_samples_split: int = 2,
                    min_samples_left: int = 1,
                    criterion: variance_reduction):

            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.criterion = criterion

        def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:

            for col in y[]
            categories = np.unique
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
