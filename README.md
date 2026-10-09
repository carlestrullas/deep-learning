# Deep Learning

Hands-on deep learning projects implemented in PyTorch. Each section is a self-contained folder with its notebook, data and a short explanation of the ideas behind it, with an emphasis on understanding *why* a technique works rather than just applying it.

## Sections

| Section | Topic | Key ideas |
|---|---|---|
| [`ordinal_classification`](ordinal_classification/) | Predicting 1–5 star hotel review ratings | Cumulative binary targets, rank-consistent outputs, order-aware evaluation (MAE, within ±1) |

## Getting started

```bash
git clone https://github.com/carlestrullas/deep-learning.git
cd deep-learning
pip install -r requirements.txt
jupyter lab
```

Each notebook loads its data with paths relative to its own folder, so open it from inside that folder.
