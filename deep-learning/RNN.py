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
    from torch.utils.data import TensorDataset, DataLoader

    return DataLoader, TensorDataset, mo, nn, np, plt, torch


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

    RNNs keep an internal state $\mathbf{{h}}$ which is passed in each step of the network.

    The forward pass looks like this:

    {mo.image(mo.notebook_dir() / "public" / "RNNforwardpass.png", alt="RNN forward pass")}

    Three sets of weights need to be learned:

    - $\mathbf{{W}}_{{xh}}$: transforms input $\mathbf{{x}}_t$ to hidden state $\mathbf{{h}}_t$
    - $\mathbf{{W}}_{{hh}}$: transforms hidden state $\mathbf{{h}}_{{t-1}}$ at previous step to next step $\mathbf{{h}}_t$
    - $\mathbf{{W}}_{{hy}}$: transforms hidden state $\mathbf{{h}}_t$ to prediction $\hat{{\mathbf{{y}}}}_t$

    With batch size $n$, $d$ input features, $h$ hidden units and $q$ outputs, the dimensions are:

    | Matrix | Shape | Meaning |
    |---|---|---|
    | $\mathbf{{X}}_t$ | $n \times d$ | inputs at time $t$ |
    | $\mathbf{{H}}_t$, $\mathbf{{H}}_{{t-1}}$ | $n \times h$ | hidden state |
    | $\mathbf{{W}}_{{xh}}$ | $d \times h$ | input → hidden |
    | $\mathbf{{W}}_{{hh}}$ | $h \times h$ | hidden → hidden |
    | $\mathbf{{b}}_h$ | $1 \times h$ | hidden bias (broadcast over the batch) |
    | $\mathbf{{W}}_{{hy}}$ | $h \times q$ | hidden → output |
    | $\mathbf{{b}}_y$ | $1 \times q$ | output bias (broadcast over the batch) |
    | $\hat{{\mathbf{{y}}}}_t$ | $n \times q$ | prediction at time $t$ |

    The forward pass looks like the following:

    Hidden state at time $t$
    $$\underbrace{{\mathbf{{H}}_t}}_{{n \times h}} = \Phi(\underbrace{{\mathbf{{X}}_t \mathbf{{W}}_{{xh}}}}_{{(n \times d)(d \times h)}} + \underbrace{{\mathbf{{H}}_{{t-1}} \mathbf{{W}}_{{hh}}}}_{{(n \times h)(h \times h)}} + \underbrace{{\mathbf{{b}}_h}}_{{1 \times h}})$$

    Output/prediction at time $t$
    $$\underbrace{{\hat{{\mathbf{{y}}}}_t}}_{{n \times q}} = \underbrace{{\mathbf{{H}}_{{t}} \mathbf{{W}}_{{hy}}}}_{{(n \times h)(h \times q)}} + \underbrace{{\mathbf{{b}}_{{y}}}}_{{1 \times q}}$$

    **Gradient Clipping**
    RNNs pass inputs in both in the input-output direction as well as in time. In each iteration gradients are backpropogated through time, meaning there is a chain of matrix operations of length $O(T)$. This many operations can cause numerical instability. The gradients can explode or vanish. 

    In RNNs, we can clip the gradients which forces the gradients to take smaller values.
    $$\mathbf{{g}} \leftarrow \min\left(1, \frac{{\Theta}}{{\lVert \mathbf{{g}} \rVert}}\right) \mathbf{{g}}$$

    This ensures that the gradient L2 norm (Euclidean distance) does not exceed $\Theta$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RNN Implementation
    """)
    return


@app.cell
def _(nn, torch):
    class RNNScratch(nn.Module):
        """A single-layer RNN, implemented from scratch.

        Computes the hidden state at each timestep:
            H_t = tanh(X_t @ W_xh + H_{t-1} @ W_hh + b_h)

        Only returns the hidden states and not the predictions.

        Args:
            num_inputs (d): number of input features per timestep 
            num_hiddens (h): number of hiddne units
            sigma: standard dev. of random starting weights

        """
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
                torch.randn(rnn.num_hiddens, num_outputs) * sigma
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
            outputs = torch.stack(outputs) # (num_steps, batch_size, num_outputs)
            return outputs, state

    def clip_gradients(params, max_norm):
        """Clips gradients by ensuring the L2 norm is not greater than the max_norm.
        Prevents the exploding gradient problem.
        """
        params = [p for p in params if p.grad is not None]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))

        if norm > max_norm:
            for p in params:
                p.grad *= max_norm / norm
        return norm

    return RNNScratch, RNNWithOutput, clip_gradients


@app.cell
def _(RNNScratch, RNNWithOutput, torch):
    # Test
    batch_size = 2
    num_inputs = 16
    num_hiddens = 32
    num_steps = 100
    num_outputs = 10

    rnn = RNNScratch(num_inputs, num_hiddens)
    model = RNNWithOutput(rnn, num_outputs)
    X = torch.ones((num_steps, batch_size, num_inputs))
    y_hat, state = model(X)

    assert y_hat.shape == (num_steps, batch_size, num_outputs)
    assert state.shape == (batch_size, num_hiddens)
    assert len(list(model.parameters())) == 5
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training the RNN
    Train the RNN on timeseries data.
    """)
    return


@app.cell
def _(torch):
    # ---------------- Settings ----------------
    T = 10000
    NOISE_STD = 0.2
    HEAT_POWER = 1.0        # heating per step when heater is on (at 0 degrees)
    MAX_TEMP = 3.0          # heating fades to zero as the room approaches this
    COOLING_RATE = 0.2      # fraction of the room-outside difference lost per step
    OUTSIDE_AMP = 1.0       # amplitude of the outside temperature cycle
    OUTSIDE_PERIOD = 200    # timesteps per outside temperature cycle
    NUM_STEPS = 20          # lookback: timesteps the model sees per window
    HORIZON = 10             # predict the temperature this many steps ahead
    TRAIN_FRACTION = 0.7
    BATCH_SIZE = 16

    NUM_HIDDENS = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.01
    CLIP_MAX_NORM = 1.0

    torch.manual_seed(0)
    return (
        BATCH_SIZE,
        CLIP_MAX_NORM,
        COOLING_RATE,
        HEAT_POWER,
        HORIZON,
        LEARNING_RATE,
        MAX_TEMP,
        NOISE_STD,
        NUM_EPOCHS,
        NUM_HIDDENS,
        NUM_STEPS,
        OUTSIDE_AMP,
        OUTSIDE_PERIOD,
        T,
        TRAIN_FRACTION,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data
    The target is a timeseries representing room temperature over time. It evolves as

    $$T_t = T_{t-1} + \underbrace{P \cdot \text{heater}_{t-1} \cdot \frac{T_{\max} - T_{t-1}}{T_{\max}}}_{\text{heating}} - \underbrace{k \, (T_{t-1} - T^{\text{out}}_{t-1})}_{\text{cooling}} + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

    - **Nonlinear:** heating gets weaker as the room approaches $T_{\max}$, so the effect of the heater depends on the current temperature.
    - **Hidden variable:** the outside temperature $T^{\text{out}}$ follows a slow sine cycle. It is **not** given to the model, which has to infer it from how the room behaves.

    The model sees two input features at each step: the current temperature and whether the heater is on.
    """)
    return


@app.cell
def _(
    COOLING_RATE,
    HEAT_POWER,
    MAX_TEMP,
    NOISE_STD,
    OUTSIDE_AMP,
    OUTSIDE_PERIOD,
    T,
    torch,
):
    # heater turns on and off randomly every 50 timesteps or so
    heater = torch.zeros(T)
    level = 0
    for t in range(T):
        if t % 50 == 0:
            level = torch.randint(0, 2, (1, ))
        heater[t] = level

    # outside temperature: a slow cycle that is NOT given to the model
    outside = OUTSIDE_AMP * torch.sin(2 * torch.pi * torch.arange(T) / OUTSIDE_PERIOD)

    temperature = torch.zeros(T)
    for t in range(1, T):
        _prev = temperature[t - 1]
        _heating = HEAT_POWER * heater[t - 1] * (MAX_TEMP - _prev) / MAX_TEMP  # weaker as the room nears MAX_TEMP
        _cooling = COOLING_RATE * (_prev - outside[t - 1])                      # heat leaks towards the outside temperature
        temperature[t] = _prev + _heating - _cooling + NOISE_STD * torch.randn(1)

    features = torch.stack([temperature, heater], dim=1) # (T, 2)
    return features, heater, outside, temperature


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Windowing and Splitting into Train/Validation
    Data is timeseries so must not be split randomly
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    DataLoader,
    HORIZON,
    NUM_STEPS,
    TRAIN_FRACTION,
    TensorDataset,
    features,
    temperature,
    torch,
):
    def make_windows(inputs, target):
        X_list = []
        Y_list = []
        for i in range(len(target) - NUM_STEPS - HORIZON + 1):
            X_list.append(inputs[i : i + NUM_STEPS])                     # (NUM_STEPS, num_features)
            Y_list.append(target[i + HORIZON : i + NUM_STEPS + HORIZON]) # (NUM_STEPS,) shifted HORIZON steps ahead

        X = torch.stack(X_list)                  # (num_windows, NUM_STEPS, num_features)
        Y = torch.stack(Y_list).unsqueeze(-1)    # (num_windows, NUM_STEPS, 1)
        return X, Y

    X_windows, Y_windows = make_windows(features, temperature)

    n_train = int(len(X_windows) * TRAIN_FRACTION)
    train_ds = TensorDataset(X_windows[:n_train], Y_windows[:n_train])
    val_ds = TensorDataset(X_windows[n_train:], Y_windows[n_train:])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
    return n_train, train_ds, train_loader, val_ds, val_loader


@app.cell(hide_code=True)
def _(
    HORIZON,
    NUM_STEPS,
    T,
    TRAIN_FRACTION,
    heater,
    outside,
    plt,
    temperature,
):
    _split = int((T - NUM_STEPS - HORIZON + 1) * TRAIN_FRACTION)  # first test window starts here

    _fig, (_ax_temp, _ax_heat) = plt.subplots(
        2, 1, figsize=(10, 4.5), sharex=True, height_ratios=[3, 1]
    )

    _ax_temp.plot(temperature, color="#2a78d6", linewidth=1.5, label="room (target)")
    _ax_temp.plot(outside, color="#1baf7a", linewidth=2, label="outside (hidden from model)")
    _ax_temp.legend(frameon=False, loc="lower left", ncols=2)
    _ax_temp.set_ylabel("Temperature")
    _ax_temp.set_title("Simulated room temperature, outside temperature and heater state")

    _ax_heat.step(range(T), heater, where="post", color="#eb6834", linewidth=1.5)
    _ax_heat.set_ylabel("Heater")
    _ax_heat.set_yticks([0, 1], ["off", "on"])
    _ax_heat.set_xlabel("Timestep")

    for _ax in (_ax_temp, _ax_heat):
        _ax.axvline(_split, color="#898781", linestyle="--", linewidth=1)
        _ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        _ax.spines[["top", "right"]].set_visible(False)

    _ax_temp.text(_split, _ax_temp.get_ylim()[1], "  validation →", va="top", color="#52514e")
    _ax_temp.text(_split, _ax_temp.get_ylim()[1], "← train  ", va="top", ha="right", color="#52514e")

    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Loop
    """)
    return


@app.cell
def _(
    CLIP_MAX_NORM,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_HIDDENS,
    RNNScratch,
    RNNWithOutput,
    clip_gradients,
    nn,
    torch,
    train_ds,
    train_loader,
    val_ds,
    val_loader,
):
    rnn_model = RNNWithOutput(
        RNNScratch(num_inputs=2, num_hiddens=NUM_HIDDENS),
        num_outputs=1
    )

    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    train_losses = []
    val_losses = []

    # early stopping: remember the weights from the epoch with the lowest validation loss
    best_val = float("inf")
    best_epoch = 0
    _best_state = None


    for _epoch in range(NUM_EPOCHS):
        _total = 0.0

        # Training
        for _X, _Y in train_loader:
            _X = _X.transpose(0, 1)     # (NUM_STEPS, batch, features)
            _Y = _Y.transpose(0, 1)     # (NUM_STEPS, batch, 1)

            _y_hat, _ = rnn_model(_X)
            _loss = loss_fn(_y_hat, _Y)

            optimizer.zero_grad()
            _loss.backward()
            clip_gradients(rnn_model.parameters(), CLIP_MAX_NORM)
            optimizer.step()

            _total += _loss.item() * _X.shape[1]

        train_losses.append(_total / len(train_ds))

        # Validation
        _total = 0.0
        with torch.no_grad():
            for _X, _Y in val_loader:
                _X, _Y = _X.transpose(0, 1), _Y.transpose(0, 1)
                _y_hat, _ = rnn_model(_X)
                _total += loss_fn(_y_hat, _Y).item() * _X.shape[1]
        val_losses.append(_total/ len(val_ds))

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_epoch = _epoch
            _best_state = {_k: _v.clone() for _k, _v in rnn_model.state_dict().items()}

        print(f"epoch {_epoch}:    train {train_losses[-1]:.4f}    val {val_losses[-1]:.4f}")

    # restore the best weights so later cells use the best model
    rnn_model.load_state_dict(_best_state)
    print(f"restored weights from epoch {best_epoch} (val {best_val:.4f})")
    return best_epoch, rnn_model, train_losses, val_losses


@app.cell
def _(best_epoch, np, plt, train_losses, val_ds, val_losses):
    _epochs = range(len(train_losses))

    # persistence baseline: predict temperature[t + HORIZON] = temperature[t] (feature 0 is temperature)
    _X_val, _Y_val = val_ds.tensors
    _persistence = ((_X_val[:, :, 0:1] - _Y_val) ** 2).mean().item()
    _marker = "o" if len(_epochs) <= 30 else None  

    _fig, _ax = plt.subplots(figsize=(9, 3.5))
    _ax.plot(_epochs, train_losses, color="#2a78d6", linewidth=2, marker=_marker, markersize=5, label="train")
    _ax.plot(_epochs, val_losses, color="#eb6834", linewidth=2, marker=_marker, markersize=5, label="validation")
    _ax.axhline(_persistence, color="#52514e", linestyle="--", linewidth=1.5, label=f"persistence baseline (val) = {_persistence:.3f}")
    _ax.axvline(best_epoch, color="#898781", linestyle=":", linewidth=1.5, label=f"best epoch = {best_epoch}")

    _lo = min(train_losses + val_losses + [_persistence]) * 0.9
    _hi = max(train_losses + val_losses + [_persistence]) * 1.1
    if _hi / _lo > 3:
        _ax.set_yscale("log")
        _decades = range(int(np.floor(np.log10(_lo))), int(np.ceil(np.log10(_hi))) + 1)
        for _steps in [(1, 2, 5), (1, 1.5, 2, 3, 4, 5, 7)]:  # use finer steps if the range is short
            _ticks = [_m * 10.0**_e for _e in _decades for _m in _steps if _lo <= _m * 10.0**_e <= _hi]
            if len(_ticks) >= 4:
                break
        _ax.set_yticks(_ticks, [f"{_t:g}" for _t in _ticks])
        _ax.minorticks_off()
        _ax.set_ylabel("MSE loss (log scale)")
    else:
        _ax.set_ylabel("MSE loss")
    _ax.set_ylim(_lo, _hi)

    _ax.xaxis.get_major_locator().set_params(integer=True)  # whole-number epochs, auto-spaced
    _ax.set_xlabel("Epoch")
    _ax.set_title("Training and validation loss")
    _ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    _ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    _ax.spines[["top", "right"]].set_visible(False)

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(X, rnn_model, torch, val_ds):
    X_val, Y_val = val_ds.tensors
    X_in = X.val.transpose(0, 1)

    with torch.no_grad():
        Y_hat, _ = rnn_model(X_in)

    val_pred = Y_hat[-1 :, 0]
    return


@app.cell
def _(HORIZON, NUM_STEPS, heater, mo, n_train, plt, rnn_model, torch, val_ds):
    # one prediction per timestep: use the LAST step of each validation window,
    # where the RNN has seen the full NUM_STEPS lookback
    _X_val, _Y_val = val_ds.tensors
    with torch.no_grad():
        _y_hat, _ = rnn_model(_X_val.transpose(0, 1))  # (NUM_STEPS, num_val_windows, 1)

    _pred = _y_hat[-1, :, 0]
    _actual = _Y_val[:, -1, 0]
    _persist = _X_val[:, -1, 0]  # temperature HORIZON steps before the target
    _t = n_train + NUM_STEPS - 1 + HORIZON + torch.arange(len(_actual))  # timestep of each target

    _SHOW = 400  # zoom into the first part of validation so individual steps are visible
    _s = slice(0, _SHOW)

    _fig, (_ax, _ax_heat) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True, height_ratios=[3, 1])
    _ax.plot(_t[_s], _actual[_s], color="#2a78d6", linewidth=1.5, label="actual")
    _ax.plot(_t[_s], _pred[_s], color="#eb6834", linewidth=2, label="RNN prediction")
    _ax.plot(_t[_s], _persist[_s], color="#898781", linewidth=1, linestyle="--", label="persistence")
    _ax.set_ylabel("Temperature")
    _ax.set_title(f"Validation: predicting {HORIZON} step(s) ahead (first {_SHOW} timesteps)")
    _ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))

    _ax_heat.step(_t[_s], heater[_t[_s] - HORIZON], where="post", color="#eb6834", linewidth=1.5)
    _ax_heat.set_ylabel("Heater\n(at input)")
    _ax_heat.set_yticks([0, 1], ["off", "on"])
    _ax_heat.set_xlabel("Timestep of target")

    for _a in (_ax, _ax_heat):
        _a.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        _a.spines[["top", "right"]].set_visible(False)

    _fig.tight_layout()

    mo.vstack([
        _fig,
        mo.md(
            f"Last-step MSE over all {len(_actual)} validation windows: "
            f"**RNN {((_pred - _actual) ** 2).mean():.3f}**, "
            f"persistence {((_persist - _actual) ** 2).mean():.3f}"
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
