# ml-notes

My machine learning notes. Updating as I find time to document what I learn.

## Contents

```
shallow-learning/
├── linear-regression.py                   # Closed form & numpy GD implementation
├── logistic-regression.py                 # numpy GD implementation
├── linear-regression-regularization.py    # numpy GD implementation for ridge, lasso and elastic regression
└── softmax-regression.py                  # numpy softmax regression implementation

deep-learning/
└── MLP.py                                 # numpy MLP classification implementation
```

## Running the Notebooks
You will need [uv](https://docs.astral.sh/uv/getting-started/installation/):

Use marimo to open and edit the notebooks.
For example:
```bash
uv run marimo edit shallow-learning/linear-regression.py
```
