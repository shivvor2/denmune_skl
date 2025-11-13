import warnings
from numbers import Integral

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context, clone
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state, get_tags
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import validate_data

REDUCER_CLASS_MAP: dict[str, BaseEstimator] = {
    "tsne": TSNE,
    "pca": PCA,
}


class DenMune(ClusterMixin, BaseEstimator):
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
        or a pre-initialized scikit-learn compatible estimator object.

    metric : str, default='euclidean'
        The distance metric to use for the k-nearest neighbor search. See
        `sklearn.neighbors.NearestNeighbors` for valid options.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

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
        input features is greater than `target_dims`. `None` if dimensionality
        reduction is not performed.

    nn_ : NearestNeighbors
        The fitted `NearestNeighbors` instance used to find the k-nearest
        neighbors of each point.

    projected_X_ : np.ndarray of shape (n_samples, target_dims)
        The input data `X` after dimensionality reduction has been applied. If
        `reduce_dims` is False, or if dimensionality is not possible, this is a
        copy of the original `X`.

    n_samples_ : int
        The number of samples in the input data `X`.

    n_features_in_ : int
        The number of features seen during :term:`fit`. Not available if
        ``metric == "precomputed"``.

    Examples
    --------
    >>> from sklearn.datasets import make_moons
    >>> import numpy as np
    >>> # Assuming DenMune is defined in the current scope or imported
    >>> from denmune_skl import DenMune
    >>>
    >>> # Generate sample data
    >>> X, y = make_moons(n_samples=200, noise=0.05, random_state=42)
    >>>
    >>> # Compute DenMune clustering
    >>> model = DenMune(k_nearest=15, random_state=42)
    >>> model.fit(X)
    DenMune(k_nearest=15, random_state=42)
    >>>
    >>> # Access the results
    >>> labels = model.labels_
    >>> n_clusters = model.n_clusters_
    >>> print(f"Estimated number of clusters: {n_clusters}")
    Estimated number of clusters: 2

    References
    ----------

    [1] Abbas, M., El-Zoghabi, A., & Shoukry, A. (2021). DenMune: Density peak based
        clustering using mutual nearest neighbors. Pattern Recognition, 109, 107589.
        https://doi.org/10.1016/j.patcog.2020.107589
    """

    _parameter_constraints: dict = {
        "k_nearest": [Interval(Integral, 1, None, closed="left")],
        "reduce_dims": [bool],
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
        X : array-like of shape (n_samples, n_features_in)
            Training instances to cluster.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
            Fitted estimator.
        """

        # 1. Input validation
        X = validate_data(self, X=X, accept_sparse=True, dtype=[np.float64, np.float32])

        self.n_samples_ = X.shape[0]

        if self.n_samples_ <= 1:
            # No reason to cluster no points or a single point
            raise ValueError(f"n_samples={self.n_samples_} should be > 1.")

        # 2. Dim reduction (if enabled)
        self.reducer_ = None

        if self.reduce_dims:
            if self.metric == "precomputed":
                raise ValueError(
                    "metric='precomputed' is not supported when reduce_dims=True"
                )

            reducer_params = {
                "n_jobs": self.n_jobs,
            }

            if hasattr(self.dim_reducer, "n_components") or self.dim_reducer in [
                "tsne",
                "pca",
            ]:
                reducer_params["n_components"] = self.target_dims

            # Special case for TSNE
            if isinstance(self.dim_reducer, str) and self.dim_reducer == "tsne":
                # Default perplexity is 30. It must be < n_samples.
                if self.target_dims > 3:
                    reducer_params["method"] = "exact"
                if self.n_samples_ <= 30:
                    reducer_params["perplexity"] = max(
                        1,
                        min(30, self.n_samples_ - 1),
                    )

            self.reducer_ = self._get_component(
                self.dim_reducer,
                REDUCER_CLASS_MAP,
                self.random_state,
                **reducer_params,
            )
            # Check Sparse compatibility
            if issparse(X) and not get_tags(self.reducer_).input_tags.sparse:
                reducer_input_tags = get_tags(self.reducer_).input_tags
                if not reducer_input_tags.sparse:
                    warnings.warn(
                        f"The selected dimensionality reducer "
                        f"({self.reducer_.__class__.__name__}) does not support sparse "
                        f"input. Skipping dimensionality reduction and proceeding with "
                        f"the original sparse data.",
                        UserWarning,
                        stacklevel=2,
                    )
            # Here metric != "precomputed" always
            elif self.n_features_in_ <= self.target_dims:
                warnings.warn(
                    f"Skipping dimensionality reduction: n_features_in_ "
                    f"({self.n_features_in_}) is not greater than target_dims "
                    f"({self.target_dims}).",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                X = self.reducer_.fit_transform(X)

        self.projected_X_ = X

        # 3. Nearest Neighbor Search
        self.nn_ = NearestNeighbors(
            n_neighbors=self.k_nearest,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        self.nn_.fit(X)
        adj_matrix = self.nn_.kneighbors_graph(X, mode="connectivity")

        # Mutual graph is the intersection of the graph and its transpose
        mutual_graph = adj_matrix.multiply(adj_matrix.T)

        # 4. Point Classification and Canonical Ordering (Unchanged)
        in_degree = np.array(adj_matrix.sum(axis=0)).flatten()
        self.density_scores_ = in_degree
        sorted_indices = np.argsort(in_degree)[::-1]

        # Get mutual neighbors for each point efficiently
        mutual_neighbors = mutual_graph.tolil().rows

        # 5. CreateClustersSkeleton (Phase I - Algorithm 2)
        # This phase builds the initial cluster backbones from strong points.

        # State-tracking arrays
        labels = np.full(self.n_samples_, -1, dtype=np.intp)
        cluster_parent = np.arange(self.n_samples_, dtype=np.intp)

        strong_point_mask = in_degree >= self.k_nearest
        self.core_sample_indices_ = np.where(strong_point_mask)[0]

        # Process strong points in canonical order to form cluster skeletons
        for i in sorted_indices:
            if not strong_point_mask[i]:
                continue

            # This point is a strong point. Mark it as initially belonging to its own
            # cluster.
            labels[i] = i

            # Per Algorithm 2, the candidate set C is the point plus its MNNs.
            # Check which existing clusters this set intersects with.
            candidate_set_indices = mutual_neighbors[i]

            # Find which of these MNNs have already been processed and assigned a
            # cluster.
            # These neighbors form the bridge to other clusters.
            classified_neighbors = [n for n in candidate_set_indices if labels[n] != -1]

            if not classified_neighbors:
                # This strong point doesn't connect to any existing cluster skeleton.
                # Keep it as a new, independent cluster for now.
                continue

            # MERGE LOGIC
            # Find the root clusters of all neighbors it connects to.
            neighbor_roots = {
                DenMune._find_root(labels[n], cluster_parent)
                for n in classified_neighbors
            }

            # Merge current point's cluster with ALL clusters it intersects.
            root_of_i = DenMune._find_root(i, cluster_parent)
            for root in neighbor_roots:
                DenMune._union(root_of_i, root, cluster_parent)

        # 6. AssignWeakPoints (Phase II - Algorithm 3)
        # This phase attaches weak points to the established skeletons.

        # Iteratively attach weak points until no more changes occur.
        while True:
            newly_assigned_count = 0
            unclassified_indices = np.where(labels == -1)[0]
            if len(unclassified_indices) == 0:
                break

            for i in unclassified_indices:
                mnn_of_i = mutual_neighbors[i]
                if not mnn_of_i:
                    continue

                classified_mnn = [n for n in mnn_of_i if labels[n] != -1]
                if not classified_mnn:
                    continue

                # Count occurrences of roots among neighbors.
                # This is equivalent to the simplified `|MNN_q_i INTERSECT C_j|` rule.
                neighbor_roots = [
                    DenMune._find_root(labels[n], cluster_parent)
                    for n in classified_mnn
                ]

                if not neighbor_roots:
                    continue

                unique_roots, counts = np.unique(neighbor_roots, return_counts=True)

                # Find the root with the maximum count (max intersection).
                # `np.argmax` handles ties deterministically by taking first occurrence.
                best_cluster_root = unique_roots[np.argmax(counts)]

                labels[i] = best_cluster_root
                newly_assigned_count += 1

            if newly_assigned_count == 0:
                break

        # 7. Setting self.labels_
        final_labels = np.full(self.n_samples_, -1, dtype=np.intp)
        classified_mask = labels != -1
        final_labels[classified_mask] = [
            DenMune._find_root(lable, cluster_parent)
            for lable in labels[classified_mask]
        ]

        # Remap the arbitrary root labels (e.g., 27, 1053, 4000) to a clean
        # 0, 1, 2... sequence
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
    def _union(i, j, parent_array):
        """Standard union operation for Union-Find."""
        root_i = DenMune._find_root(i, parent_array)
        root_j = DenMune._find_root(j, parent_array)
        if root_i != root_j:
            parent_array[root_j] = root_i

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
            valid_params = component_class._get_param_names()
            # valid_params = component_class.get_params().keys()
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
                        f"the following unsupported arguments are supplied "
                        f"{rejected_kwargs}",
                        category=UserWarning,
                        stacklevel=2,
                    )
                instance.set_params(**filtered_kwargs)

            return instance

    def __sklearn_tags__(self):
        # Get the parent tags
        tags = super().__sklearn_tags__()
        # Define custom tags
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = False
        tags.target_tags.required = False
        return tags
