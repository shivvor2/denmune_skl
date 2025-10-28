import numpy as np
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator, ClusterMixin, _fit_context
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
    Real,
    Integral,
    Boolean,
)

from scipy.sparse import csr_matrix


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
        Indices of core samples, i.e., the initial cluster centers.
    """

    # TODO: Add proper validation, check TSNE class definition for hints
    _parameter_constraints: dict = {
        "k_nearest": [Interval(Integral, 1, None, closed="left")],
        "reduce_dims": [Boolean],
        "target_dims": [Interval(Integral, 1, None, closed="left")],
        "dim_reducer": [StrOptions({"tsne", "pca", "umap"}), BaseEstimator],
        "dim_reducer_args": [dict, None],
        "metric": [str],
        "n_jobs": [Integral, None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        k_nearest=10,
        reduce_dims=True,
        target_dims=2,
        dim_reducer="tsne",
        dim_reducer_args=None,
        metric="euclidean",
        n_jobs=None,
        random_state=None,
    ):
        self.k_nearest = k_nearest
        self.reduce_dims = reduce_dims
        self.target_dims = target_dims
        self.dim_reducer = dim_reducer
        self.dim_reducer_args = dim_reducer_args
        self.metric = metric
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
        # TODO: Check what types of sparse inputs we can take

        # In fit, at the beginning:
        X = validate_data(X, accept_sparse=True, dtype=[np.float64, np.float32])
        # X = check_array(X, accept_sparse=True, dtype=[np.float64, np.float32], estimator = self)

        # 2. Dimensionality Reduction (if applicable)
        self.n_samples = X.shape[0]

        # Store original version of X (?)

        if self.reduce_dims:
            # What should I pass in as class args and what should I pass in via self.dim_reducer_args?
            # Raise warning when args in self.dim_reducer_args override object variables
            reducer = TSNE(
                n_components=self.target_dims, n_jobs=self.n_jobs
            )  # TODO: Replace with actural selection arguments
            X = reducer.fit_transform(X)

        # 3. Nearest Neighbor Search (vectorized)
        # TODO: Supply NearestNeighbors Args to constructor
        nn = NearestNeighbors(n_neighbors=self.k_nearest + 1, n_jobs=self.n_jobs)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        # 4. Homogeneity Calculation (vectorized)
        # TODO: Allow swapping of the homogenety metric, we are only replicating the thing here

        # indices[:, 1:] because we skip the point itself
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
        reference_counts = np.array(
            mutual_graph.sum(axis=1)
        ).flatten()  # mutual neighbors

        homogeneity = 100 * referred_by_counts + reference_counts

        self.homogeneity_scores_ = homogeneity

        # 5. prepare_Clusters (using Union-Find)

        # State-tracking arrays
        labels = np.full(self.n_samples, -1, dtype=np.intp)
        cluster_parent = np.arange(self.n_samples, dtype=np.intp)

        # Sort points by homogeneity to find potential cluster centers ("Kernel Points")
        # Iterate from most dense to least dense
        sorted_indices = np.argsort(homogeneity)[::-1]

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
        check_is_fitted(self)  # Should be called after fit()
        return self.labels_

    @staticmethod
    def _find_root(i, parent_array):
        if parent_array[i] == i:
            return i
        # Path compression for optimization
        parent_array[i] = DenMune._find_root(parent_array[i], parent_array)
        return parent_array[i]
