import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


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

    With batch size $n$, $d$ input features, $h$ hidden units and $q$ outputs, the dimensions are:

    | Matrix | Shape | Meaning |
    |---|---|---|
    | $X_t$ | $n \times d$ | inputs at time $t$ |
    | $H_t$, $H_{{t-1}}$ | $n \times h$ | hidden state |
    | $W_{{xh}}$ | $d \times h$ | input → hidden |
    | $W_{{hh}}$ | $h \times h$ | hidden → hidden |
    | $b_h$ | $1 \times h$ | hidden bias (broadcast over the batch) |
    | $W_{{hy}}$ | $h \times q$ | hidden → output |
    | $b_y$ | $1 \times q$ | output bias (broadcast over the batch) |
    | $\hat{{y}}_t$ | $n \times q$ | prediction at time $t$ |

    The forward pass looks like the following:

    Hidden state at time $t$
    $$\underbrace{{H_t}}_{{n \times h}} = \Phi(\underbrace{{X_t W_{{xh}}}}_{{(n \times d)(d \times h)}} + \underbrace{{H_{{t-1}} W_{{hh}}}}_{{(n \times h)(h \times h)}} + \underbrace{{b_h}}_{{1 \times h}})$$

    Output/prediction at time $t$
    $$\underbrace{{\hat{{y}}_t}}_{{n \times q}} = \underbrace{{H_{{t}} W_{{hy}}}}_{{(n \times h)(h \times q)}} + \underbrace{{b_{{y}}}}_{{1 \times q}}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RNN Implementation
    """)
    return


app._unparsable_cell(
    """
    class RNNScratch(nn.Module):
        \"\"\"A single-layer RNN, implemented from scratch.

        Computes the hidden state at each timestep:
            H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b_h)

        Only returns the hidden states and not the predictions.

        Args:
            num_inputs (d): number of input features per timestep 
            num_hiddens (h): number of hiddne units
            sigma: standard dev. of random starting weights
    
        \"\"\"
        def __init__(self, num_inputs, num_hiddens, sigma=0.01):
            super().__init__()
            self.num_inputs = num_inputs
            self.num_hiddens = num_hiddens
            self.sigma = sigma

            self.W_xh = nn.Parameter(
                torch.randn(num_inputs, num_hiddens) * sigma
            )
            self.W_hh = nn.Parameter(
                torch.randn(num_hiddens, num_hiddens) * sigma
            )
            self.b_h = nn.Parameter(
                torch.zeros(num_hiddens)
            )

        def forward(self, inputs, state=None):
            # inputs shape: (num_steps, batch_size, num_inputs)
            if state is None:
                batch_size = inputs.shape[1]
                state = torch.zeros(batch_size, self.num_hiddens, device = inputs.device)

            outputs = []
            for X in inputs:
                state = torch.tanh(X @ self.W_xh + state @ self.W_hh + self.b_h)
                outputs.append(state)
            return outputs, state

    class RNNWithOutput(nn.Module):
        def __init__(self, rnn, num_outputs, sigma=0.01):
            super().__init__()
            self.rnn = rnn
            self.num_outputs = num_outputs

            self.W_hy = nn.Parameter(
                torch.randn(rnn.num_hiddens, num_outputs)
            )

            self.b_y = nn.Parameter(
                torch.zeros(num_outputs)
            )

        def forward(self, inputs, state=None):
            hiddens, state = self.rnn(inputs, state)

            outputs = []
            for H in hiddens:
                y_hat = H @ self.W_hy + self.b_y
                outputs.append(y_hat)
                return outputs, state

        def 
    """,
    name="_"
)


@app.cell
def _(RNNScratch, torch):
    num_steps = 100
    batch_size = 2
    num_inputs = 16
    num_hiddens = 32

    model = RNNScratch(num_inputs, num_hiddens)
    X = torch.ones(num_steps, batch_size, num_inputs)
    outputs, state = model(X)

    print(len(outputs))     # 100
    print(outputs[0].shape) # torch.Size([2, 32])
    print(state.shape)      # torch.Size([2, 32])
    return


if __name__ == "__main__":
    app.run()
