import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_iris, make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB as SkGaussianNB

    return load_iris, mo, plt, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Naive Bayes Classifier
    A probabilistic classifier. The goal is calculate the probability an example is of a certain class given the features $P(Y=y|X)$.

    We get calcualte this probability using Bayes' Theorem

    $$P(Y|X) = \frac{P(X|Y)P(Y)}{P(X)}$$

    ## Likelihood $P(X|Y)$
    Assumes that all features ($X_1, X_2, ...,X_n$) are independent. Meaning that all $P(X|Y=y)$ are independent events that do not affect each other. This means we can calculate the likelihood as:
    $$P(X|Y) = \prod_{i=1}^{n} P(X_i | Y)$$

    For continuous features we assume they are normally distributed. Therefore, $P(X_i=x_i|Y=y)$ is given by the Gaussian probability density function.
    $$P(X_i = x_i \mid Y = y) = \frac{1}{\sigma_{y,i}\sqrt{2\pi}} \exp\left( -\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2} \right)$$

    ## Prior $P(Y)$
    The prior is just how often each class appears in the training set:
    $$P(Y = y) = \frac{n_y}{n}$$

    ## Evidence $P(X)$
    Is constant and therefore not calculated.

    ## Predictions
    We know that:
    $$P(Y|X) \propto P(Y) \prod_{i=1}^{n} P(X_i | Y)$$
    Thus, predictions are given by:
    $$\hat{y} = \argmax_{y}P(Y) \prod_{i=1}^{n} P(X_i | Y)$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementation
    """)
    return


app._unparsable_cell(
    r"""
    class NaiveBayes:

        def fit(self, X: np.ndarray, y: np.ndarray):
            self.classes = np.unique(y)
            self.priors = [len(y[y == c]) / len(y) for c in self.classes]

            self.means = [X[y == c].mean() for c in self.classes]
            self.stds = [X[y == c].std() for c in self.classes]

        def compute_likelihood(self, row, class_idx): 
        
    """,
    name="_"
)


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
def _(X, train_test_split, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    return


if __name__ == "__main__":
    app.run()
