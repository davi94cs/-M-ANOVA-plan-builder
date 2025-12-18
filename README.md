# -M-ANOVA-plan-builder (v1.0.0-alpha)

## Overview

This module builds an **(M)ANOVA experimental grid under a strict computational time budget**.
The goal is to maximize the **statistical coverage of model families** while ensuring
that the total execution time remains feasible.

The workflow is divided into **two main steps**:

1. **Construction of an execution-time table** (`time_table.tsv`)  
   → estimates runtime scalability and extrapolates execution times to larger datasets.
2. **Construction of the ANOVA plan** (`anova_plan_Xh.tsv`)  
   → selects an optimal subset of models using combinatorial optimization under
   a fixed time budget (`X` hours).

This approach is suited for **large-scale ML benchmarking and ANOVA-based analyses**,
where exhaustive evaluation is computationally prohibitive.

---

## Step 1 — Build the execution time table

**Script:** `build_time_table.py`  
**Output:** `time_table.tsv`

This step generates a table containing **measured and estimated execution times**
for all combinations of:

- feature selection method
- classifier
- sample distribution
- dataset size

### Output schema

| Column | Description |
|------|-------------|
| `fs_method` | Feature selection method |
| `classifier` | Classifier name |
| `α` | Scalability exponent |
| `dataset_size` | Number of samples |
| `ex_time_sec` | Estimated execution time (seconds) |
| `ex_time_h` | Estimated execution time (hours) |

---

### Runtime scaling model

Let:

- \( N \) = number of samples  
- \( c \) = constant  
- \( \alpha \) = scalability exponent  
- \( T(N) \) = execution time for input size \( N \)

We assume a power-law scaling:

\[
T(N) = c \cdot N^{\alpha}
\]

Given two observations \((T_1, N_1)\) and \((T_2, N_2)\):

\[
\alpha = \frac{\log(T_2 / T_1)}{\log(N_2 / N_1)}
\]

---

### α estimation strategy

In practice, α is estimated using **two empirical measurements**:

- \( (T_{80}, N_{80}) \): runtime with an 80/20 split
- \( (T_{70}, N_{70}) \): runtime with a 70/30 split

Since dataset size scales with training fraction:

\[
\alpha = \frac{\log(T_{80} / T_{70})}{\log(0.8 / 0.7)}
\]

The exponent is:
- estimated per `(fs_method, classifier)` pair when possible
- clipped to a predefined range
- otherwise replaced by the **global median α**

---

### Execution time extrapolation

Once α is known, execution times are extrapolated from the base dataset
(typically 3k samples) to larger datasets.

For example, estimating runtime on 24k samples:

\[
T(24k) = T(3k) \cdot 8^{\alpha}
\]

The final `time_table.tsv` contains execution-time estimates for:

- 500 samples
- 3,000 samples
- 24,000 samples

---

## Step 2 — Build the ANOVA plan

**Script:** `build_anova_plan.py`  
**Output:** `anova_plan_<X>h.tsv` (where `X` is the time budget in hours)

This step selects a **subset of classifiers** that:

- respects a global execution-time budget
- includes **at most one classifier per model family**
- maximizes the number of represented families

This is formulated as a **combinatorial optimization problem**.

---

### Model families

Classifiers are grouped into families to avoid redundancy:

```python
family_map = {
  "CAT": "boosting", "HGB": "boosting", "XGB": "boosting",
  "RF": "bagging", "ET": "bagging",
  "DT": "tree",
  "LR": "linear",
  "LDA": "discriminant_linear",
  "GNB": "naive_bayes",
  "KNN": "knn",
  "MLP": "mlp",
  "SVC": "kernel_svm"
}
```

### Optimization Problem

### 1. Decision variables

- **Binary decision variable** \( X_m \):
  - \( X_m = 1 \) if model *m* is selected
  - \( X_m = 0 \) otherwise

---

### 2. Constraints

- **Time budget**  
  The total execution time of all selected models must not exceed
  a predefined budget (e.g. 100h, 500h).

- **Family constraint**  
  At most **one classifier per model family** can be selected.

- **Design completeness**  
  If a model is selected, **all its experimental combinations**
  must be included in the ANOVA design:
  - feature selection method
  - dataset size
  - sample distributions

- **Replications**  
  If `n_dist > 1`, the total execution time is multiplied accordingly.

---

### 3. Objective

- **Maximize the number of selected model families**

---

### Optimization Strategy

The optimization problem is solved via **Depth-First Search (DFS)**,
which is equivalent to a **multiple-choice knapsack problem**.

For each model family, the algorithm considers two alternatives:
- select **no model** from the family
- select **exactly one model** from the family

During the search:
- branches exceeding the time budget are **pruned early**
- only feasible solutions are explored

Each complete solution is evaluated according to:
- the number of selected model families
- the total execution time

The **globally optimal solution is guaranteed** because:
- the number of model families is limited
- each family has only a few candidate classifiers
- DFS explores the full feasible search space with effective pruning

### Output Schema

The  (M)ANOVA plan is saved as a **tsv file** with the following columns:

| Column | Description |
|------|-------------|
| `fs_method` | Feature selection method |
| `classifier` | Selected classifier |
| `dataset_size` | Dataset size |
| `distribution` | Sample distribution |
| `ex_time_h` | Estimated execution time (hours) |

Where:

- `fs_method`, `classifier` `dataset_size`, `distribution` are the independent (explainatory) variables of (M)ANOVA plan;
- `ex_time_h` is the dependent (response) variable of (M)ANOVA plan

---

### Allowed Values

- **`fs_method`** ∈ `["enSFM", "laSFM", "none", "rfSFM", "GWAS"]`

- **`classifier`** ∈  
  `["CAT", "DT", "ET", "GNB", "HGB", "KNN", "LDA", "LR", "MLP", "RF", "SVC", "XGB"]`

- **`dataset_size`** ∈ `["500", "3000", "24000"]`

- **`distribution`** ∈ `["1", "2", "3"]`

---

### Future steps

- Include `accuracy` as response variable and `n_CPUs` as explainatory variable.
- Update and improve the builder.





