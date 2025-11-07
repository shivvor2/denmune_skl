"""
===================================================================
DenMune with a Custom Dimensionality Reducer (UMAP)
===================================================================

DenMune's `dim_reducer` parameter is flexible, allowing you to pass not just
pre-defined strings ('tsne', 'pca') but also any scikit-learn compatible
estimator instance.

This example shows how to use UMAP (`umap-learn`) as the dimensionality
reducer. This is particularly useful as UMAP is often faster than t-SNE and can
be better at preserving global data structure.

Note: To run this example, you must have the `umap-learn` library installed.
`pip install umap-learn`
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Try to import UMAP. If it fails, skip the example.
try:
    from umap import UMAP
except ImportError:
    print("UMAP not found. Skipping this example.")
    # sphinx-gallery will not run the rest of the script if it exits with code 0
    import sys

    sys.exit(0)

from denmune_skl import DenMune

# %%
# Generate high-dimensional data
# ------------------------------
# We create a high-dimensional dataset that requires reduction to be clustered
# effectively by DenMune's 2D-focused approach.
X_high_dim, y = make_blobs(
    n_samples=500, n_features=50, centers=5, cluster_std=2.5, random_state=42
)

# %%
# Initialize DenMune with a UMAP instance
# -----------------------------------------------
# We create an instance of `UMAP` and pass it directly to the `dim_reducer`
# parameter of DenMune. DenMune will then use this instance for its
# dimensionality reduction step.

umap_reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)

model = DenMune(
    k_nearest=30,
    reduce_dims=True,
    target_dims=2,  # This will be ignored, but is a required parameter
    dim_reducer=umap_reducer,
    random_state=42,
)

labels = model.fit_predict(X_high_dim)

# %%
# Visualize the results on the projected data
# -------------------------------------------
# The clustering is performed on the 2D data projected by UMAP. We can access
# this projected data via the `projected_X_` attribute.

X_projected = model.projected_X_

plt.figure(figsize=(10, 8))

# Plot projected data colored by DenMune labels
plt.subplot(1, 2, 1)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=labels, s=20, cmap="viridis")
plt.title(f"DenMune Labels (k={model.k_nearest})\n(Projected by UMAP)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")

# Plot projected data colored by true labels for comparison
plt.subplot(1, 2, 2)
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, s=20, cmap="viridis")
plt.title("True Labels\n(Projected by UMAP)")
plt.xlabel("UMAP Component 1")

plt.suptitle("DenMune with Custom UMAP Reducer")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
