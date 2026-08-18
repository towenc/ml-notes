# ml-notes

My machine learning notes. Updating as I find time to document what I learn.

## Contents

### shallow-learning/
linear-regression.py                   - Closed form & numpy GD implementation
logistic-regression.py                 - numpy GD implementation
linear-regression-regularization.py    - numpy GD implementation for ridge, lasso and elastic regression
softmax-regression.py                  - numpy softmax regression implementation
decision-trees.py                      - numpy regression & classification decision tree implementation
random-forest.py                       - numpy regression & classification random forest implementation

###deep-learning/
[MLP.py](https://molab.marimo.io/github/github.com/towenc/ml-notes/blob/main/deep-learning/MLP.py)                               - numpy MLP classification implementation

## Running the Notebooks
You will need [uv](https://docs.astral.sh/uv/getting-started/installation/):

Use marimo to open and edit the notebooks.
For example:
```bash
uv run marimo edit shallow-learning/linear-regression.py
```
Alternatively, just click the hyperlinks in contents and notebooks will run on molab! 
