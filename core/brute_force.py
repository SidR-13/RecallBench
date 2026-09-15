import numpy as np


class BruteForce:
    """Exact k-nearest-neighbor search by checking every vector.

    This is the ground-truth oracle recall is measured against everywhere
    else in this project — deliberately independent of HNSW's internals
    (see core/hnsw.py) so a shared bug can't hide from both sides agreeing
    with each other. Distance metric is squared L2, matching HNSW exactly,
    since recall is only meaningful when both sides rank by the same
    metric.
    """

    def __init__(self, vectors, ids=None):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        if ids is None:
            self.ids = np.arange(len(self.vectors))
        else:
            self.ids = np.asarray(ids)
            if len(self.ids) != len(self.vectors):
                # This class is the ground-truth oracle every recall
                # number in this project is measured against (see class
                # docstring) -- a caller-side off-by-one that zips the
                # wrong id to the wrong vector must fail loudly here, not
                # surface later as a confusing IndexError from
                # self.ids[candidate_idx] or, worse, silently return a
                # real-looking but wrong id for a mismatched-length ids
                # array that happens not to index out of bounds.
                raise ValueError(
                    f"ids length ({len(self.ids)}) does not match vectors "
                    f"length ({len(self.vectors)})"
                )

    def __len__(self):
        return len(self.vectors)

    def search(self, query, k):
        """Exact k-NN for one query.

        Returns list[(node_id, distance)], ascending by distance, length
        <= k. Distance is squared L2, same convention as HNSW.search().
        """
        query = np.asarray(query, dtype=np.float32)

        # Squared L2 to every stored vector in one vectorized pass, no
        # Python-level loop over the corpus.
        diffs = self.vectors - query
        dists = np.einsum("ij,ij->i", diffs, diffs)

        n = len(dists)
        k = min(k, n)
        if k == 0:
            return []

        if k == n:
            # Need every distance ranked anyway — argpartition buys nothing.
            candidate_idx = np.arange(n)
        else:
            # argpartition finds the k smallest in O(n), unordered; only
            # those k then get ranked below (O(k log k)) instead of ranking
            # all n.
            candidate_idx = np.argpartition(dists, k)[:k]

        # Rank the k candidates by (distance, id) instead of distance alone.
        # Ties on distance are otherwise left to argpartition/argsort's
        # implementation-defined tie order — not wrong, but not
        # reproducible, and "ground truth" has to be reproducible run to
        # run. lexsort's LAST key is primary, so (ids, dists) sorts by
        # dists first, ids second.
        order = np.lexsort((self.ids[candidate_idx], dists[candidate_idx]))
        top_idx = candidate_idx[order]

        # .item() converts any numpy scalar back to its native python type
        # (np.str_ -> str, np.int64 -> int, ...) -- ids aren't assumed to be
        # numeric, they mirror HNSW's arbitrary node_id.
        return [(self.ids[i].item(), float(dists[i])) for i in top_idx]
