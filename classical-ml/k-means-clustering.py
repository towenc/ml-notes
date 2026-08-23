import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from numpy.random import default_rng
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans as SkKMeans

    return SkKMeans, make_blobs, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # K-Means Clustering
    Unsupervised machine learning method that groups data into k clusters.

    **The algorithm (Lloyd's algorithm) is described as follows:**

    1. Randomly initalise k centroids
    2. Repeat until convergence:
        1. Assign each point to the nearest centroid. $c(i) = \arg\min_j \lVert x_i - \mu_j \rVert^2$.]
        2. If centroid has no points assigned to it, randomly assign another point as the centroid.
        3. Update each centroid to the mean of the points that has been assigned to it  $\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x_i$.

    Each step minimizes the cost function which is the inertia (within cluster sum of squares)
    $$J = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementation

    Fill in the bodies below. Suggested order: `_assign` -> `_update` -> `_inertia`
    -> `_lloyd` -> `fit` -> `_init_centroids` (random first, k-means++ after).
    """)
    return


@app.cell
def _(np):
    class KMeans:
        """K-means clustering using Lloyd's algorithm

        Parameters:
            n_clusters: int, default=6
                Number of clusters 'k' to group the data into.
                Must not exceed n_samples
            max_inter: int, default=300
                Maximum number of iterations to run fit loop.
            tol: float, default=1e-4
                Convergence threshold on centroid movement between iterations
            random_state: int or None, default=None
                Seed for centroid initalization and empty-cluster reseeding.


        Attributes:
            centroids_: ndarray of shape (n_clusters, n_features)
                Final centroid positions set by fit.
            labels_: ndarray of shape (n_samples,)
                Cluster index for each data point in [0, n_clusters)
            inertia_: float
                Final value of the cost function J
            n_iter_: int
                Interations run before convergence or max_itewr  
        """
        def __init__(self, n_clusters = 6, max_iter=300, tol=1e-4, random_state=None):
            self.n_clusters = n_clusters
            self.max_iter = max_iter
            self.tol = tol
            self.random_state = random_state

        def _init_centroids(self, X, k, rng):
            """Randomly initalises k centroids.
            """
            idx = rng.choice(len(X), size=k, replace=False)
            return X[idx].copy()

        def _sq_dist(self, X, centroids):
            """Calculates the Euclidean distance from every point to every centroid.

            Parameters:
                X: ndarray of shape (n_samples, n_features)
                centroids:: ndarray of shape (n_clusters, n_features)

            Returns:
                ndarray of shape(n_samples, n_clusters)
            """
            diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
            return (diff**2).sum(axis=-1)

        def _assign(self, X, centroids):
            """Assigns each data point a centroid. Each point is assigned to the closest by centroid
               by Euclidean distance.
            """
            return self._sq_dist(X, centroids).argmin(axis=1)

        def _update_centroids(self, X, labels, k, rng):
            """Moves each centroid to the mean of the points assigned to it.

            Parameters:
                X: ndarray of shape (n_samples, n_features)
                    Data points
                labels: ndarray of shape (n_samples)
                    Cluster index of each point [0, k)
                k: int
                    Number of clusters
                rng: numpy.random.Generator
                    Passed in. Used for reinitalizing centroids with no points assigned to it

            Returns:
                ndarray of shape (n_clusters, n_features)
                    Updated centroid positions
            """
            d = X.shape[1]
            centroids = np.empty((k, d), dtype=X.dtype)
        
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = X[mask].mean(axis=0)
                else:
                    # if there are no members in the centroid then assign centroid to random point
                    centroids[j] = X[rng.integers(len(X))]
            return centroids

        def fit(self, X):
            """Lloyd's algorithm is run until centroid positions converge

            Parameters:
            X: ndarray of shape (n_samples, n_features)

            Returns:
                self: KMeans
                    Fitted estimator
            """
            X = np.asarray(X, dtype=float)
            k = self.n_clusters
            if k > len(X):
                raise ValueError(f"n_clusters={k} exceeds n_samples={len(X)}")

            rng = np.random.default_rng(self.random_state)
            centroids = self._init_centroids(X, k, rng)

            n_iter = 0
            for n_iter in range(self.max_iter + 1):
                labels = self._assign(X, centroids)

                old_centroids = centroids
                centroids = self._update_centroids(X, labels, k, rng)

                shift = ((old_centroids - centroids) ** 2).sum()

                if shift <= self.tol:
                    break

            self.centroids_ = centroids
            self.labels_ = self._assign(X, centroids)

            # Sum the distances from assigned centroid across all data samples. (J, the cost func.)
            self.inertia_ = float(self._sq_dist(X, centroids).min(axis=1).sum())
            self.n_iter_ = n_iter
            return self

    return (KMeans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison
    """)
    return


@app.cell
def _(make_blobs, plt):
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.9)

    X_blobs.shape
    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap="tab10", s=20)
    return X_blobs, y_blobs


@app.cell
def _(KMeans, SkKMeans, X_blobs, mo):
    km = KMeans(n_clusters=4, random_state=0).fit(X_blobs)
    sk = SkKMeans(n_clusters=4, n_init=1, random_state=0).fit(X_blobs)

    mo.md(
        f"""
        **From scratch Cost:** {km.inertia_:.4f}\n
        **sklearm Cost**:     {sk.inertia_:.4f}
        """
    )
    return km, sk


@app.cell
def _(X_blobs, km, plt, sk, y_blobs):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)

    panels = [
        ("ground truth", y_blobs, None),
        (f"from scratch  (inertia={km.inertia_:.1f})", km.labels_, km.centroids_),
        (f"sklearn  (inertia={sk.inertia_:.1f})", sk.labels_, sk.cluster_centers_),
    ]

    for ax, (title, labels, centers) in zip(axes, panels):
        ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels,
                   cmap="tab10", s=18, alpha=0.8)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=250,
                       c="black", edgecolors="white", linewidths=1.5, zorder=3)
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Selecting K
    Use an elbow plot to see how many clusters could be ideal for your data.
    """)
    return


@app.cell
def _(KMeans, X_blobs, plt):
    ks = range(1, 11)
    inertias = [KMeans(n_clusters=k, random_state=0).fit(X_blobs).inertia_ for k in ks]
    plt.plot(ks, inertias, "o-")
    plt.xlabel("No. of clusters K")
    plt.ylabel("Inertia")
    plt.title("Elbow Plot")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this case, elbow plot is showing 4 as the ideal number of clusters for this data.
    """)
    return


if __name__ == "__main__":
    app.run()
