# Coreset Selection with Feature Distance and Quality Weighting

This repository provides a PyTorch-based implementation of a greedy **coreset selection** algorithm.
The method incrementally selects samples that maximize diversity in feature space, optionally weighted by a per-sample quality score.

## Overview

Given:
- A set of feature vectors
- (Optionally) a quality score for each sample

The algorithm:
1. Starts from an initial sample (index `0`)
2. Iteratively selects the next sample that maximizes the objective function

Typical use cases:
- Dataset subset selection
- Diversity-aware sampling
- Quality-aware coreset construction

## Usage

```python
from coreset import execute_coreset_selection

selected_files = execute_coreset_selection(
    feature_dict_path="diversity.pt",
    quality_dict_path="nisqa.pt",
    alpha=8,
    size=4000,
    device=None  # automatically uses CUDA if available
)
```

## Function Arguments

| Argument            | Type                     | Description                                             |
| ------------------- | ------------------------ | ------------------------------------------------------- |
| `feature_dict_path` | `str`                    | Path to a `.pt` file containing `{key: feature_tensor}` |
| `quality_dict_path` | `str`                    | Path to a `.pt` file containing `{key: quality_score}`  |
| `alpha`             | `int`                    | Quality exponent. If `0`, quality is ignored            |
| `size`              | `int`                    | Number of samples to select                             |
| `device`            | `torch.device` or `None` | Computation device                                      |

## Return Value

* `List[str]`: Keys corresponding to the selected samples, in the order they were selected.

## Input Format
### Feature Dictionary

```python
{
    "file_001": torch.Tensor([...]),
    "file_002": torch.Tensor([...]),
    ...
}
```

All feature tensors must have the same shape.

### Quality Dictionary (`nisqa.pt`)

```python
{
    "file_001": float,
    "file_002": float,
    ...
}
```

All keys must match those in the feature dictionary.

## License

MIT License
