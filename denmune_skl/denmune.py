import warnings
from typing import Any

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
    Real,
    Integral,
    Boolean,
)
from scipy.sparse import csr_matrix, issparse

REDUCER_CLASS_MAP: dict[str, BaseEstimator] = {
    "tsne": TSNE,
    "pca": PCA,
}


class DenMune(BaseEstimator, ClusterMixin):
    """
    DenMune: A density-peak based clustering algorithm.

    This implementation is a refactored and optimized version of the original
    algorithm published in Pattern Recognition (2021). It adheres to the
    scikit-learn API and offers significant performance improvements.

    Parameters
    ----------
    k_nearest : int, default=10
        The number of nearest neighbors to use for density estimation. This is
        the primary parameter of the algorithm.

    reduce_dims : bool, default=True
        If True, performs dimensionality reduction to `target_dims` before
        clustering. Recommended for high-dimensional data.

    target_dims : int, default=2
        The number of dimensions to reduce the data to if `reduce_dims` is True.

    dim_reducer : str or estimator object, default='tsne'
        The dimensionality reduction method to use. Can be 'tsne', 'pca',
        'umap' (if installed), or a pre-initialized scikit-learn compatible
        estimator object.

    dim_reducer_params : dict, default = None
        Arguments provided to the inner dimension reducer object, ignored if
        `dim_reducer` is an estimator object

    metric : str, default='euclidean'
        The distance metric to use for the k-nearest neighbor search. See
        `sklearn.neighbors.NearestNeighbors` for valid options.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search and
        dimensionality reduction. `None` means 1, `-1` means using all
        processors.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noise points are labeled -1.

    core_sample_indices_ : np.ndarray of shape (n_core_samples,)
        Indices of core samples. In the context of DenMune, these are the
        "Strong Points" that have an in-degree of at least `k_nearest` in the
        k-NN graph and are used to form the initial cluster skeletons.

    n_clusters_ : int
        The estimated number of clusters found by the algorithm. This does
        not include the noise cluster if one exists.

    density_scores_ : np.ndarray of shape (n_samples,)
        The density score calculated for each point. In this implementation,
        this corresponds to the in-degree (`|KNN_p<-|`) from the k-NN graph,
        which is used for the "canonical ordering" step of the algorithm.

    reducer_ : estimator object
        The fitted dimensionality reduction estimator instance. This attribute
        is only set if the `reduce_dims` parameter is True and the number of
        input features is greater than `target_dims`.

    nn_ : NearestNeighbors
        The fitted `NearestNeighbors` instance used to find the k-nearest
        neighbors of each point.

    projected_X_ : np.ndarray of shape (n_samples, target_dims)
        The input data `X` after dimensionality reduction has been applied. If
        `reduce_dims` is False, this is a copy of the original `X`.

    n_samples_ : int
        The number of samples in the input data `X`.

    n_features_in_ : int
        The number of features seen during :term:`fit`. Not availiable if
        `metric == "precomputed".

    Examples
    --------
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.preprocessing import StandardScaler
    >>> import numpy as np
    >>> # Assuming DenMune is defined in the current scope or imported
    >>> # from denmune import DenMune
    >>>
    >>> # Generate sample data
    >>> X, y = make_moons(n_samples=250, noise=0.07, random_state=42)
    >>> X = StandardScaler().fit_transform(X)
    >>>
    >>> # Compute DenMune clustering
    >>> model = DenMune(k_nearest=15, random_state=42)
    >>> model.fit(X)
    >>>
    >>> # Access the results
    >>> labels = model.labels_
    >>> n_clusters = model.n_clusters_
    >>> print(f"Estimated number of clusters: {n_clusters}")
    Estimated number of clusters: 2
    >>>
    >>> # Visualize the results
    >>> import matplotlib.pyplot as plt
    >>>
    >>> core_samples_mask = np.zeros_like(labels, dtype=bool)
    >>> core_samples_mask[model.core_sample_indices_] = True
    >>> unique_labels = set(labels)
    >>>
    >>> fig, ax = plt.subplots(figsize=(8, 6))
    >>> colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    >>>
    >>> for k, col in zip(unique_labels, colors):
    ...     if k == -1:
    ...         # Black used for noise.
    ...         col = [0, 0, 0, 1]
    ...
    ...     class_member_mask = labels == k
    ...
    ...     # Plot core samples
    ...     xy = X[class_member_mask & core_samples_mask]
    ...     ax.plot(
    ...         xy[:, 0],
    ...         xy[:, 1],
    ...         "o",
    ...         markerfacecolor=tuple(col),
    ...         markeredgecolor="k",
    ...         markersize=12,
    ...         label=f"Cluster {k} (Core)" if k != -1 else "Noise",
    ...     )
    ...
    ...     # Plot non-core samples (boundary points)
    ...     xy = X[class_member_mask & ~core_samples_mask]
    ...     ax.plot(
    ...         xy[:, 0],
    ...         xy[:, 1],
    ...         "o",
    ...         markerfacecolor=tuple(col),
    ...         markeredgecolor="k",
    ...         markersize=6,
    ...         label=f"Cluster {k} (Boundary)" if k != -1 else "",
    ...     )
    ...
    >>> ax.set_title(f"DenMune Clustering (k={model.k_nearest})\nEstimated clusters: {n_clusters}")
    >>> ax.set_xlabel("Feature 1")
    >>> ax.set_ylabel("Feature 2")
    >>> # Create a legend that doesn't duplicate labels
    >>> handles, labels = plt.gca().get_legend_handles_labels()
    >>> by_label = dict(zip(labels, handles))
    >>> plt.legend(by_label.values(), by_label.keys())
    >>> plt.show()

    References
    ----------
    .. [1] Abbas, M., El-Zoghabi, A., & Shoukry, A. (2021). DenMune: Density peak based
        clustering using mutual nearest neighbors. Pattern Recognition, 109, 107589.
        https://doi.org/10.1016/j.patcog.2020.107589
    """

    _parameter_constraints: dict = {
        "k_nearest": [Interval(Integral, 1, None, closed="left")],
        "reduce_dims": [Boolean],
        "target_dims": [Interval(Integral, 1, None, closed="left")],
        "dim_reducer": [StrOptions({"tsne", "pca"}), BaseEstimator],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        k_nearest=10,
        reduce_dims=True,
        target_dims=2,
        dim_reducer="tsne",
        metric="euclidean",
        metric_params=None,
        n_jobs=None,
        random_state=None,
    ):
        self.k_nearest = k_nearest
        self.reduce_dims = reduce_dims
        self.target_dims = target_dims
        self.dim_reducer = dim_reducer
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.random_state = random_state

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        """
        Perform DenMune clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """
        # 1. Input validation

        X = validate_data(X, accept_sparse=True, dtype=[np.float64, np.float32])
        # X.shape is always 2d because `validate_data` prevents nd data (n >= 3)

        # Check for incompatible combinations of hyperparameters

        if self.metric == "precomputed":
            self.n_samples_ = X.shape[0]  # No information on amount of features
        else:
            self.n_samples_, self.n_features_in_ = X.shape

        # 2. Dim reduction (if enabled)
        if self.reduce_dims:
            if self.metric == "precomputed":
                raise ValueError(
                    "Cannot perform dimensionality reduction when metric is 'precomputed'. "
                    "Set reduce_dims=False or provide a feature matrix."
                )
            if self.n_features_in_ <= self.target_dims:
                warnings.warn(
                    f"Skipping dimensionality reduction: n_features ({self.n_features_in_}) "
                    f"is not greater than target_dims ({self.target_dims}).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self.reducer_ = self._get_component(
                    self.dim_reducer,
                    REDUCER_CLASS_MAP,
                    self.random_state,
                    n_components=self.target_dims,
                    n_jobs=self.n_jobs,
                )
                if issparse(X):
                    reducer_tags = self.reducer_._get_tags()
                    if "sparse" not in reducer_tags.get("X_types", []):
                        raise ValueError(
                            f"The selected dimensionality reducer ({self.reducer_.__class__.__name__}) "
                            "does not support sparse matrix input. To process sparse data, "
                            "set reduce_dims=False or provide a sparse-compatible reducer "
                            "instance (e.g., UMAP from 'umap-learn')."
                        )
                X = self.reducer_.fit_transform(X)

        self.projected_X_ = X

        # 3. Nearest Neighbor Search
        self.nn_ = NearestNeighbors(
            n_neighbors=self.k_nearest + 1,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        self.nn_.fit(X)
        distances, indices = self.nn_.kneighbors(X)

        # 4. Graph Construction and Point Classification
        adj_matrix = csr_matrix(
            (
                np.ones(self.n_samples * self.k_nearest),
                (
                    np.arange(self.n_samples).repeat(self.k_nearest),
                    indices[:, 1:].flatten(),
                ),
            ),
            shape=(self.n_samples, self.n_samples),
        )

        # Mutual graph is the intersection of the graph and its transpose
        mutual_graph = adj_matrix.multiply(adj_matrix.T)

        # Calculate homogeneity score
        referred_by_counts = np.array(adj_matrix.sum(axis=0)).flatten()  # in-degree
        self.density_scores_ = referred_by_counts
        reference_counts = np.array(
            mutual_graph.sum(axis=1)
        ).flatten()  # mutual neighbors

        # 5. prepare_Clusters (using Union-Find)
        # State-tracking arrays
        labels = np.full(self.n_samples, -1, dtype=np.intp)
        cluster_parent = np.arange(self.n_samples, dtype=np.intp)

        # Sort points by density score (in-degree) as per paper's "Canonical ordering"
        sorted_indices = np.argsort(self.density_scores_)[::-1]

        self.core_sample_indices_ = []

        # Get the indices of the mutual neighbors for each point from the sparse matrix
        mutual_neighbors = mutual_graph.tolil().rows

        for i in sorted_indices:
            # Condition to be a core point (density peak)
            # len(refer_to) is always self.k_nearest
            is_strong_enough = referred_by_counts[i] >= self.k_nearest
            has_references = reference_counts[i] > 0

            if not (is_strong_enough and has_references):
                continue

            # This point is a potential cluster center. Mark it as such.
            # Its initial label is its own index.
            labels[i] = i
            self.core_sample_indices_.append(i)

            # --- Voting and Merging Logic ---
            # Get neighbors of point `i` that are in the mutual graph
            neighbors_of_i = mutual_neighbors[i]

            # Find which of these neighbors have already been assigned a cluster
            # (i.e., they were processed earlier in this loop)
            classified_neighbors = [n for n in neighbors_of_i if labels[n] != -1]

            if not classified_neighbors:
                continue  # No classified neighbors to merge with, it's a new cluster.

            # Find the root clusters of these neighbors
            neighbor_roots = [
                DenMune._find_root(labels[n], cluster_parent)
                for n in classified_neighbors
            ]

            # VOTE: Find the most common root cluster among neighbors
            unique_roots, counts = np.unique(neighbor_roots, return_counts=True)
            majority_root = unique_roots[np.argmax(counts)]

            # UNION: Merge all neighbor clusters and the current point's cluster
            # into the majority root cluster.
            root_of_i = DenMune._find_root(labels[i], cluster_parent)
            cluster_parent[root_of_i] = majority_root
            for root in unique_roots:
                if root != majority_root:
                    cluster_parent[root] = majority_root

        # 6. attach_Points (vectorized)
        # The original code has two phases (strong and weak). We can simplify this.
        # We iterate until no more points can be attached.
        while True:
            # Find all unclassified points that have at least one mutual neighbor
            unclassified_mask = (labels == -1) & (reference_counts > 0)

            if not np.any(unclassified_mask):
                break  # No more points to attach

            points_to_attach_indices = np.where(unclassified_mask)[0]

            newly_assigned_count = 0
            for i in points_to_attach_indices:
                # Get mutual neighbors and find which are classified
                neighbors_of_i = mutual_neighbors[i]
                classified_neighbors = [n for n in neighbors_of_i if labels[n] != -1]

                if not classified_neighbors:
                    continue

                # Vote for the majority cluster
                neighbor_roots = [
                    DenMune._find_root(labels[n], cluster_parent)
                    for n in classified_neighbors
                ]
                unique_roots, counts = np.unique(neighbor_roots, return_counts=True)
                majority_root = unique_roots[np.argmax(counts)]

                # Assign the point to this cluster
                labels[i] = majority_root
                newly_assigned_count += 1

            if newly_assigned_count == 0:
                break  # No progress was made in this iteration, stop.

        # 7. Set self.labels_

        final_labels = np.full(self.n_samples, -1, dtype=np.intp)
        classified_mask = labels != -1
        final_labels[classified_mask] = [
            DenMune._find_root(lable, cluster_parent)
            for lable in labels[classified_mask]
        ]

        # Remap the arbitrary root labels (e.g., 27, 1053, 4000) to a clean 0, 1, 2... sequence
        unique_final_labels = np.unique(final_labels[final_labels != -1])
        self.n_clusters_ = len(unique_final_labels)

        label_map = {
            old_label: new_label
            for new_label, old_label in enumerate(unique_final_labels)
        }

        # Apply the mapping. Noise points (-1) are unaffected.
        self.labels_ = np.array(
            [label_map.get(lable, -1) for lable in final_labels], dtype=np.intp
        )
        self.core_sample_indices_ = np.array(self.core_sample_indices_, dtype=np.intp)

        return self

    def fit_predict(self, X, y=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        accessing the `labels_` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, y).labels_

    @staticmethod
    def _find_root(i, parent_array):
        if parent_array[i] == i:
            return i
        # Path compression for optimization
        parent_array[i] = DenMune._find_root(parent_array[i], parent_array)
        return parent_array[i]

    @staticmethod
    def _get_component(component, default_map, random_state, **kwargs):
        """
        Builds a component instance from a string or clones a user-provided instance.
        """
        if isinstance(component, str):
            try:
                component_class = default_map[component]
            except KeyError:
                # Should be caught by _validate_params, but this is a safeguard.
                raise ValueError(
                    f"Invalid string identifier for component: {component}"
                )

            # Construct parameters for the default instance
            instance_params = kwargs.copy()

            # Only pass parameters that the component class actually accepts
            valid_params = component_class.get_param_names()
            final_params = {
                k: v for k, v in instance_params.items() if k in valid_params
            }

            # Get new random seed for Child Estimator from rng to prevent correlation
            if "random_state" in valid_params:
                random_state = check_random_state(random_state)
                final_params["random_state"] = random_state.randint(
                    np.iinfo(np.int32).max  # Capped to int32
                )

            return component_class(**final_params)

        else:  # estimator instance
            # Clone the estimator to get unfitted instance w/ same params
            instance = clone(component)
            # The user is responsible for configuring their passed-in object.
            try:
                instance.set_params(**kwargs)
            except ValueError:
                # Filter kwargs to only what the instance accepts
                valid_params = instance.get_params().keys()
                filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                # Raise warning if re reject kwargs
                rejected_kwargs = set(kwargs.keys()) - set(valid_params)
                if rejected_kwargs:
                    warnings.warn(
                        f"the following unsupported arguments are supplied {rejected_kwargs}",
                        category=UserWarning,
                        stacklevel=2,
                    )
                instance.set_params(**filtered_kwargs)

            return instance
