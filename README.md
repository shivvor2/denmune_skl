# DenMune-Sklearn: A Robust Implementation of the DenMune Clustering Algorithm

[![Python package](https://github.com/shivvor2/denmune-skl/actions/workflows/python-app.yml/badge.svg)](https://github.com/shivvor2/denmune-skl/actions/workflows/python-app.yml)
[![Deploy documentation](https://github.com/shivvor2/denmune-skl/actions/workflows/deploy-gh-pages.yml/badge.svg)](https://shivvor2.github.io/denmune-skl/)
[![PyPI version](https://badge.fury.io/py/denmune-skl.svg)](https://badge.fury.io/py/denmune-skl) <!-- Change package name to sklearn-contrib-denmune when accepted (?) -->
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

An efficient, scikit-learn compatible implementation of the "DENMUNE: Density peak based clustering using mutual nearest neighbors" algorithm.

**Disclaimer:** This project is a clean-room rewrite of the DenMune algorithm. It is not derived from the original authors' code and makes significant implementation choices to improve performance and scikit-learn compatibility. It is not a strictly faithful reproduction of the paper's original pseudocode but adheres to its core principles.

---

### Why DenMune-Sklearn?

The original DenMune paper presents a powerful algorithm for finding clusters of arbitrary shapes and densities. However, existing third-party implementations suffer from critical issues, including a lack of scikit-learn compatibility, data leakage bugs, and severe performance bottlenecks.

This project was created to provide a version of DenMune that is:
- **Scikit-Learn Native:** Fully compatible with the scikit-learn ecosystem (`Pipeline`, `GridSearchCV`, etc.).
- **Correct & Robust:** Fixes fundamental algorithmic flaws like data leakage and provides robust error handling.
- **Performant:** Replaces slow Python loops with optimized, vectorized NumPy operations and a sparse graph representation.

### Key Features

- **Finds Complex Clusters:** Excels at identifying non-convex clusters of varying densities.
- **Automatic Noise Detection:** Automatically identifies and labels noise points.
- **Single Parameter Tuning:** Requires only one primary parameter, `k_nearest`, for straightforward tuning.
- **Efficient Implementation:** Uses a sparse `csr_matrix` for k-NN graph representation and a Union-Find data structure for fast cluster merging.
- **Flexible Dimensionality Reduction:** Integrates with scikit-learn's `TSNE` and `PCA`, and allows for any user-provided reducer (e.g., UMAP).

### Installation

```bash
pip install denmune-sklearn
```

### Quick Usage

```python
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from denmune import DenMune
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate and prepare data
X, y = make_moons(n_samples=250, noise=0.07, random_state=42)
X = StandardScaler().fit_transform(X)

# 2. Fit the model
model = DenMune(k_nearest=20, random_state=42)
model.fit(X)

# 3. Visualize the results
labels = model.labels_
n_clusters = model.n_clusters_

print(f"Estimated number of clusters: {n_clusters}")

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title(f"DenMune Clustering (k={model.k_nearest})\nEstimated clusters: {n_clusters}")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

### Reference

This algorithm is based on the following paper:
```
@article{Abbas2021DenMune,
  title   = {{DENMUNE}: {Density} peak based clustering using mutual nearest neighbors},
  author  = {Abbas, Mohamed and El-Zoghabi, Adel and Shoukry, Amin},
  journal = {Pattern Recognition},
  volume  = {112},
  pages   = {107718},
  year    = {2021},
  doi     = {10.1016/j.patcog.2020.107718}
}
```

### License

This project is licensed under the BSD 3-Clause License.
