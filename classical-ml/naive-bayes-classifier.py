import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_iris, load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB as SkGaussianNB

    return SkGaussianNB, load_breast_cancer, mo, np, train_test_split


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


@app.cell
def _(np):
    class NaiveBayes:
        """Gaussian Naive Bayes classifier.

        Attributes:
        ------------
        classes: ndarray (n_classes,)
            Unique class labels.
        priors: list of float, length n_classes
            Class prior probabilities, estimated as frequency of each class in training set
        means: list of ndarray, each with shape (n_features,)
            Per-class, per-feature sample means
        stds: list of ndarray, each with shape (n_features,)
            Per-class, per-feature sample standard deviation
        """
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Estimates class priors and per-features mean and standard deviations

            Parameters
            -----------
            X : ndarray of shape (n_samples, n_features)
                Training samples. All features are treated as continuous
            y : ndarray of shape (n_samples,)
                class label for each sample

            Returns
            --------
            self : NaiveBayes
            """
            self.classes = np.unique(y)
            self.priors = [len(y[y == c]) / len(y) for c in self.classes]

            self.means = [X[y == c].mean(axis=0) for c in self.classes]
            self.stds = [X[y == c].std(axis=0) for c in self.classes]

            return self

        def _compute_likelihood(self, row, class_idx):
            """ Computes the class-condition likelihood of one sample.

            Parameters
            ----------
            row : ndarray of shape (n_features,)
                A single sample
            class_idx : int
                Position of the class in self.classes

            Returns
            --------
            likelihood : float
                Product of per-feature Gaussian probability densities.
            """
            likelihood = 1

            for feature in range(len(row)):
                mean = self.means[class_idx][feature]
                std = self.stds[class_idx][feature]
                likelihood *= (1/(np.sqrt(2*np.pi) * std)) * np.exp(-((row[feature]-mean)**2)/(2 * std**2))
            return likelihood

        def predict(self, X):
            """Predicts the class label for each sample.

            For every sample, class probabilities are computed by prior * likelihood and the argmax is returned

            Parameters
            ----------
            X : ndarray of shape (n_samples, n_features)
                Samples to classify

            Returns
            --------
            y_pred : ndarray of shape (n_samples,)
                Predicted labels
            """
            y_pred = []
            for row in X:
                posteriors = []
            
                for i in range(len(self.classes)):
                    likelihood = self._compute_likelihood(row, i)
                    posteriors.append(likelihood * self.priors[i])

                y_pred.append(self.classes[np.argmax(posteriors)])

            return np.array(y_pred)

    return (NaiveBayes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Helper Functions
    """)
    return


@app.cell
def _(np):
    def confusion_matrix(y_true, y_pred):
        """
        Computes the confusion matrix
        """
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        return TP, FP, TN, FN

    def metrics(TP, FP, TN, FN):
        """
        Computes accuracy, precision, recall and f1
        """
        accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return accuracy, precision, recall, f1

    def report(y_true, y_pred, title="Results"):

        TP, FP, TN, FN = confusion_matrix(y_true, y_pred)
        acc, prec, rec, f1 = metrics(TP, FP, TN, FN)
    
        print(f"\n{title}")
        print("=" * 38)
        print(f"{'':10}{'pred 0':>10}{'pred 1':>10}")
        print(f"{'actual 0':10}{TN:>10}{FP:>10}")
        print(f"{'actual 1':10}{FN:>10}{TP:>10}")
        print("-" * 38)
        for name, val in [("accuracy", acc), ("precision", prec), ("recall", rec), ("f1", f1)]:
            print(f"{name:14}{val:>10.4f}")

    return (report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fitting and Comparison
    """)
    return


@app.cell
def _(load_breast_cancer, train_test_split):
    # Load the breast cancer dataset.
    breast = load_breast_cancer()
    X, y, names, classes = breast.data, breast.target, list(breast.feature_names), breast.target_names
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    return Xte, Xtr, yte, ytr


@app.cell
def _(NaiveBayes, SkGaussianNB, Xte, Xtr, report, yte, ytr):
    # From scratch implementation
    nb = NaiveBayes()
    nb.fit(Xtr, ytr)
    predictions = nb.predict(Xte)

    # Sk learn implementation
    sknb = SkGaussianNB()
    sknb.fit(Xtr, ytr)
    sk_predictions = nb.predict(Xte)

    report(predictions, yte, title="From Scratch")
    report(sk_predictions, yte, title="Sklearn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From scratch implementation and Sklearn implementation match exactly.
    """)
    return


if __name__ == "__main__":
    app.run()
