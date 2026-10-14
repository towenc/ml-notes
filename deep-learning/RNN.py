import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn

    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recurrent Neural Network (RNN)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    RNNs pass information from one step of the network to the next so **information persists** in the network. In other words, RNN can take into account the past to predict the future. Therefore, they are used in prediction of sequences such as in timeseries, natural language or audio.

    {mo.image(mo.notebook_dir() / "public" / "unrolledRNN.png", alt="RNN unrolled over time")}

    RNNs keep an internal state $h$ which is passed in each step of the network.

    The forward pass looks like this:

    {mo.image(mo.notebook_dir() / "public" / "RNNforwardpass.png", alt="RNN forward pass")}

    Three sets of weights need to be learned:

    - $W_{{xh}}$: transforms input $x_t$ to hidden state $h_t$
    - $W_{{hh}}$: transforms hidden state $h_{{t-1}}$ at previous step to next step $h_t$
    - $W_{{hy}}$: transforms hidden state $h_t$ to prediction $\hat{{y}}_t$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Forward**

    TODO: recurrence for $h_t$ and the output

    **Backward through time**

    TODO: why the gradient reaching $h_k$ is a product of Jacobians, and what makes it vanish / explode
    """)
    return


@app.cell
def _():
    def init_params(d, h, C, seed=42):
        """
        Initalizes the RNN weights.
        Tip: nn.RNN uses U(-1/sqrt(h), 1/sqrt(h)); matching it makes the comparison below easy.

        Parameters
        -----------
        d: int
            Number of input features per time step
        h: int
            Hidden state size
        C: int
            Number of classes
        seed: int
            Seed for rng

        Returns
        -----------
        params: dict
            "W_xh" (d, h), "W_hh" (h, h), "b_h" (h,), "W_hy" (h, C), "b_y" (C,)
            Tensors with requires_grad=True so autograd tracks them.
        """
        raise NotImplementedError

    def rnn_forward(X, params):
        """
        Forward pass of the RNN, unrolled over time.

        Parameters
        ----------
        X : tensor, shape (m, T, d)
            Batch of m sequences, T steps, d features per step.
        params : dict
            As returned by init_params.

        Returns
        -------
        logits : tensor, shape (m, C)
            Output computed from the final hidden state h_T.
        hs : list of tensor, length T
            Hidden states h_1..h_T, each (m, h). Kept so we can look at
            the gradient reaching each step.
        """
        raise NotImplementedError

    def train_rnn(X, y, h=32, lr=1e-2, epochs=300, seed=42):
        """
        Full-batch training. Gradients come from autograd.

        Returns
        -------
        params : dict
            Trained weights.
        history : list of float
            Cost at each epoch.
        """
        raise NotImplementedError

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison with PyTorch
    """)
    return


@app.cell
def _():
    # TODO: copy your params into nn.RNN(..., batch_first=True) and check the hidden states match
    # hint: nn.RNN stores weights as (out, in) and has two biases (bias_ih_l0, bias_hh_l0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create Data

    A long-memory task: each sequence is $T$ random $\pm 1$ values, and the label is the sign of the **first** one. The other $T-1$ steps are distractors from the same distribution, so the network has to hold one bit across all $T$ steps and ignore everything after it.
    """)
    return


@app.cell
def _(torch):
    def make_sequences(m, T, seed=0):
        """
        Returns X (m, T, 1) of random +-1 and y (m,) = (X[:, 0, 0] > 0).
        """
        g = torch.Generator().manual_seed(seed)
        X = torch.randint(0, 2, (m, T, 1), generator=g).float() * 2 - 1
        y = (X[:, 0, 0] > 0).long()
        return X, y

    T = 100
    X_train, y_train = make_sequences(1000, T, seed=0)
    X_val, y_val = make_sequences(500, T, seed=1)
    return


@app.cell
def _():
    # TODO: an LSTM classifier with the same interface as rnn_forward: returns (logits, hs)
    # hint: nn.LSTM only returns its outputs, so to get dJ/dh_t for every step,
    # step an nn.LSTMCell manually
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vanishing Gradients
    """)
    return


@app.cell
def _():
    # TODO: at init, backprop the loss and plot ||dJ/dh_t|| against t for the RNN and the LSTM
    # hint: h.retain_grad() keeps .grad on non-leaf tensors; a log y-axis helps
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training: RNN vs LSTM
    """)
    return


@app.cell
def _():
    # TODO: train both on X_train, report accuracy on X_val
    return


@app.cell
def _():
    # TODO: plot both loss curves, with chance level ln(2) as a reference line
    return


if __name__ == "__main__":
    app.run()
