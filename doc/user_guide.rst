.. _user_guide:

==========
User Guide
==========

This guide provides a detailed overview of the DenMune algorithm, its implementation
details, and practical usage tips.

.. contents::
   :local:

Motivation for this Implementation
----------------------------------
This version of DenMune was developed as a clean-room rewrite to provide a robust,
performant, and scikit-learn compatible estimator. The previous implementation
available on PyPI had several architectural issues that hindered its practical use,
including:

- **Lack of Scikit-learn Compatibility**: The legacy class did not inherit from
  `BaseEstimator`, preventing its use in standard scikit-learn tools like
  ``Pipeline`` and ``GridSearchCV``. Our implementation, :class:`~denmune_skl.DenMune`,
  is a proper scikit-learn estimator.
- **Algorithmic Flaws**: The previous version improperly mixed training and testing data,
  leading to data leakage and invalid evaluation metrics. This implementation adheres to
  the standard `fit`/`predict` paradigm.
- **Performance Bottlenecks**: Core logic relied on slow Python loops. This implementation
  uses a vectorized, sparse-graph approach for significant speed-up.

How DenMune Works
-----------------
DenMune is a density-based algorithm that identifies clusters by finding density
peaks and expanding from them based on mutual nearest neighbors. It operates in
two main phases after an initial density estimation.

**1. Density Estimation**

For each point `p`, the algorithm computes two key sets based on its `k_nearest` neighbors:
- The **Refer-To List (`KNN_p->`)**: The `k` points closest to `p`.
- The **Reference-List (`KNN_p<-`)**: The set of points that consider `p` to be one of their `k` nearest neighbors.

The size of the Reference-List, `|KNN_p<-|`, serves as the density score for point `p`.

**2. Phase I: Cluster Skeleton Construction**

Points are classified based on their density:
- **Strong Points**: Points where `|KNN_p<-| >= k_nearest`. These are density peaks and
serve as the initial seeds for clusters.
- **Weak Points**: Points where `|KNN_p<-| < k_nearest`. These are boundary points or noise.
- **Noise Points**: Points with few or no mutual neighbors.

The algorithm iterates through the strong points in descending order of density. Each strong
point forms a new cluster or is merged with an existing cluster skeleton using a
Union-Find data structure for efficiency. Merging is decided by a majority vote
among its already-classified mutual neighbors.

**3. Phase II: Weak Point Assignment**

After the skeletons are formed, the algorithm iteratively attempts to assign the
remaining weak points. A weak point is assigned to the cluster that has the
majority representation among its mutual neighbors. Points that cannot be assigned
are labeled as noise (`-1`).

Practical Usage and Parameters
------------------------------

**Choosing `k_nearest`**

The `k_nearest` parameter is the most important hyperparameter. It controls the
granularity of the density estimation.
- A **small `k`** may cause dense clusters to be fragmented.
- A **large `k`** may cause distinct but close clusters to be merged.
The paper suggests the algorithm is stable over a wide range of `k`. You can
investigate this for your dataset as shown in our :ref:`sphx_glr_auto_examples_plot_k_sensitivity.py` example.

**Dimensionality Reduction**

The algorithm's density estimation is most effective in low-dimensional space. For
high-dimensional data (`n_features > 50` is a common heuristic), it is highly
recommended to use dimensionality reduction.
- **`reduce_dims=True`** (default): Will reduce data to `target_dims`.
- **`dim_reducer`**: Can be set to `'tsne'`, `'pca'`, or a custom estimator instance
like UMAP. See the :ref:`sphx_glr_auto_examples_plot_custom_reducer.py` example for
how to use a custom reducer.
