import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_moons
    import torch
    import torch.nn as nn

    return make_moons, mo, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multilayer Perceptron (MLP)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A feedforward neural network with fully connected layers and non-linear activation functions.

    **Forward**

    $$Z^{[l]} = A^{[l-1]}W^{[l]} + b^{[l]}, \qquad l = 1,\dots,L$$

    $$A^{[l]} = \phi\left(Z^{[l]}\right), \qquad l = 1,\dots,L-1$$

    $$P = \operatorname{softmax}\left(Z^{[L]}\right)$$

    $$J = -\frac{1}{m}\sum_{n=1}^{m}\sum_{k=1}^{C} y_{nk}\log p_{nk}$$

    **Backward — base case**

    $$\frac{\partial J}{\partial Z^{[L]}} = \frac{1}{m}\left(P - Y\right)$$

    **Backward — recurrence, $l = L-1,\dots,1$**

    $$\frac{\partial J}{\partial A^{[l]}} = \frac{\partial J}{\partial Z^{[l+1]}}\left(W^{[l+1]}\right)^{\top}$$

    $$\frac{\partial J}{\partial Z^{[l]}} = \frac{\partial J}{\partial A^{[l]}} \odot \phi'\left(Z^{[l]}\right)$$

    **Parameter gradients, all $l$**

    $$\frac{\partial J}{\partial W^{[l]}} = \left(A^{[l-1]}\right)^{\top}\frac{\partial J}{\partial Z^{[l]}}$$

    $$\frac{\partial J}{\partial b^{[l]}} = \sum_{n=1}^{m}\frac{\partial J}{\partial z^{[l]}_{n}}$$

    **Update**

    $$W^{[l]} \leftarrow W^{[l]} - \alpha\frac{\partial J}{\partial W^{[l]}}, \qquad b^{[l]} \leftarrow b^{[l]} - \alpha\frac{\partial J}{\partial b^{[l]}}$$

    **ReLU**

    $$\phi(z) = \max(0, z), \qquad \phi'(z) = \mathbb{1}\left[z > 0\right]$$
    """)
    return


@app.cell
def _(np):
    def init_params(sizes, seed=42):
        """
        Initalizes the weights and bias' for the MLP
        Weights drawn from standard normal whereas bias' are initalized as 0

        Parameters
        -----------
        sizes: list of int
            Layer depths from input to output. [d, h1, ..., hn, C]
            d is number of features
            C is number of classes
            h is number of neurons in each hidden layer
        seed: int
            Seed for rng

        Returns
        -----------
        params: dict
            Keys "W1", ... ,"WL", "b1", ..., "bL" where L is len(sizes)-1
            W has shape (sizes[l-1],size[l])
            b has shape (sizes[l], )
        """
        rng = np.random.default_rng(seed)
        params = {}
        for l in range(1, len(sizes)):    
            n_in = sizes[l -1]
            n_out = sizes[l]
            W = rng.standard_normal((n_in, n_out))
            b = np.zeros(n_out)

            params[f"W{l}"] = W

            params[f"b{l}"] = b
        return params

    def relu(Z):
        """
        Applies ReLU
        """
        return np.maximum(0.0, Z)

    def softmax(Z):
        """
        Applies softmax
        """
        Z = Z - np.max(Z, axis=1, keepdims=True) 
        f = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        return f

    def forward(X, params):
        """
        Forward propagation through the network.

        ReLU is applied to every hidden layer. Output layer is left as raw
        logits and passed through softmax. Intermediates are returned because
        backward needs them: A^{l-1} for the weight gradients, Z^{l} for the
        ReLU.

        Parameters
        ----------
        X : ndarray, shape (m, n)
            Input batch. m examples, n features.
        params : dict
            Weights and biases as returned by init_params.

        Returns
        -------
        P : ndarray, shape (m, C)
            Class probabilities. Rows sum to 1.
        As : list of ndarray, length L
            Layer activations. As[0] is X, so As[l-1] is the input to layer l.
            The final activation is not stored here -- it is P.
        Zs : list of ndarray, length L
            Pre-activations. Zs[l-1] is layer l's, so Zs[-1] holds the logits.
        """
    
        L = len(params) // 2 # Number of layers
        A = X
        Zs = []
        As = [X]
        for l in range(1, L+1):
            Z = A @ params[f"W{l}"] + params[f"b{l}"]
            Zs.append(Z)
            # leaves last hidden layer for softmax activation
            if l < L:
                A = relu(Z)
                As.append(A)
    
        P = softmax(Zs[-1]) 
        return P, As, Zs

    def compute_loss(P, y):
        """
        Calculates cross-entropy loss
        """
        m = len(y)
        # Takes out probability of the true class
        p_hat = P[np.arange(m), y]
        # 1e-12 added to avoid log(0)
        L = -np.log(p_hat + 1e-12)
        return L

    def compute_cost(P, y):
        """
        Takes the mean over the dataset of loss to find the cost.
        """
        return np.mean(compute_loss(P, y))

    def backward(y, P, As, Zs, params):
        """
        Backpropagation through the network.

        Assumes ReLU on the hidden layers and softmax on the output. The softmax
        and cross-entropy derivatives are fused, which is why the base case is
        simply (P - Y) / m rather than two separate steps.

        Walks layers from L down to 1. At each layer it reads off that layer's
        weight and bias gradients, then -- unless it is at layer 1 -- carries
        dJ/dZ back one layer by crossing W and then the ReLU. dJ_dA and dJ_dZ are
        partial products of the chain to help compute dJ_w[l]

        Parameters
        ----------
        y : ndarray, shape (m,)
            Integer class labels in [0, C).
        P : ndarray, shape (m, C)
            Class probabilities from forward.
        As : list of ndarray, length L
            Layer activations from forward. As[l-1] is the input to layer l.
        Zs : list of ndarray, length L
            Pre-activations from forward (wx+b). Zs[l-1] is layer l's.
        params : dict
            Weights and biases.

        Returns
        -------
        grads : dict
            Same keys and shapes as params, so the update rule is
            params[k] -= lr * grads[k] for every k.
        """
        m = len(y)
        L = len(params) // 2
        grads = {}

        # one hot encode the output
        y = np.eye(P.shape[1])[y]
        # output layer
        dJ_dZ = (P - y) / m

        # for loop goes backwards through the layers
        for l in range(L, 0, -1):
            grads[f"W{l}"] = As[l-1].T @ dJ_dZ
            grads[f"b{l}"] = np.sum(dJ_dZ, axis=0)

            # Update dJ_dA to update dJ_dZ
            if l > 1:
                dJ_dA = dJ_dZ @ params[f"W{l}"].T

                # I want Z from layer (l-1)
                # Zs holds layers 1..L at indices 0..L-1
                # thereform layer k is at Zs[k-1] -> (l-1-1)
                dJ_dZ = dJ_dA * (Zs[l-2] > 0)
        return grads

    def train(X, y, sizes, lr=0.1, epochs=200, seed=42):
        """
        Train the network with full-batch gradient descent.

        Parameters
        ----------
        X : ndarray, shape (m, d)
            Training inputs.
        y : ndarray, shape (m,)
            Integer class labels in [0, C).
        sizes : list of int
            Layer widths from input to output, e.g. [d, h1, ...,hn, C].
        lr : float, optional
            Learning rate. 
        epochs : int, optional
            Number of gradient steps.
        seed : int, optional
            Seed passed to init_params.

        Returns
        -------
        params : dict
            Trained weights and biases.
        history : list of float
            Cost at each epoch.
        """
        params = init_params(sizes, seed=seed)
        history = []

        for epoch in range(epochs):
            P, As, Zs = forward(X, params)
            history.append(compute_cost(P,y))

            grads = backward(y, P, As, Zs, params)
            for k in params:
                params[k] -= lr * grads[k]

        return params, history

    return forward, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison with PyTorch
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Data
    """)
    return


@app.cell
def _(make_moons, np, plt):
    X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
    X = X.astype(np.float64)

    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Pastel1", edgecolor="k", s=25)
    plt.title("make_moons, noise=0.2")
    plt.show()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Fitting
    """)
    return


@app.cell
def _(X, forward, nn, torch, train, y):
    ## Implemented Approach
    sizes = [2, 8, 2]
    params, history_gd = train(X, y, sizes=sizes, lr=0.5, epochs=500)
    P, _, _ = forward(X, params)

    print("Implementation accuracy:", (P.argmax(axis=1) == y).mean())

    ## PyTorch
    torch.manual_seed(42)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    history_torch = []
    for epoch in range(500):
        logits = model(Xt)
        loss = criterion(logits, yt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history_torch.append(loss.item())
    
    with torch.no_grad():
        preds = model(Xt).argmax(dim=1)
        acc = (preds == yt).float().mean().item()

    print(f"PyTorch accuracy: {acc:.4f}")
    return model, params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison
    """)
    return


@app.cell
def _(X, forward, model, np, params, plt, torch, y):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 300),
                         np.linspace(X[:,1].min()-.5, X[:,1].max()+.5, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # numpy implementation
    zz_np = forward(grid, params)[0][:, 1].reshape(xx.shape)

    # torch
    with torch.no_grad():
        gt = torch.tensor(grid, dtype=torch.float32)
        zz_pt = torch.softmax(model(gt), dim=1)[:, 1].numpy().reshape(xx.shape)

    for ax, zz, title in [(axes[0], zz_np, "from scratch"),
                          (axes[1], zz_pt, "pytorch")]:
        ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.7)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
        ax.scatter(X[:,0], X[:,1], c=y, cmap="RdBu", edgecolor="k", s=20)
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
