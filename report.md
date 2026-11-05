---
title: Homework 1 -- Your Title Here
author: John Doe
course: Artificial Intelligence · UCU
date: 2026-10-12
---

## Introduction

Write your report here using regular **Markdown**. This text is an example — replace it.

## Method

Inline math works with single dollars: $f(x) = \sigma(Wx + b)$. Display math uses double dollars:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} y_i \log \hat{y}_i
$$

Code blocks are highlighted:

```python
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()
```

## Results

| Model    | Accuracy | F1   |
|----------|----------|------|
| Baseline | 0.71     | 0.68 |
| Ours     | 0.84     | 0.82 |

Images go into the `images/` folder:

<!-- ![Training loss](images/loss.png) -->

## Conclusion

> Summarize your findings here.
