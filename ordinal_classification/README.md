# Ordinal Classification

Star ratings, severity levels and grades are ordered, but a standard softmax classifier treats them as unrelated categories: predicting 1★ for a 5★ review costs the same as predicting 4★. This section shows how to build neural network classifiers that respect the order between classes.

## What's inside

The notebook [`ordinal_classification.ipynb`](ordinal_classification.ipynb) compares three models built on the same PyTorch backbone:

- **Standard classifier:** softmax over the 5 ratings, trained with cross-entropy.
- **Ordinal classifier:** 4 cumulative binary outputs $P(y > k)$, trained with binary cross-entropy on "thermometer" targets.
- **Rank-consistent ordinal classifier:** a single shared score with 4 strictly decreasing thresholds, so the cumulative probabilities can never contradict each other.

The models are evaluated with accuracy, mean absolute error in stars, the share of predictions within ±1 star, and the rate of inconsistent cumulative probabilities, together with learning curves, confusion matrices and error-size distributions.

## Data

| File | Content |
|---|---|
| `data/review_data.pickle` | DataFrame with 7,105 hotel reviews (`Review`) and their 1–5 star `Rating`, balanced across classes |
| `data/embeddings.pickle` | NumPy array `(7105, 768)` of precomputed text embeddings, aligned row by row with the reviews |

## Running it

```bash
pip install -r ../requirements.txt
jupyter lab ordinal_classification.ipynb
```
