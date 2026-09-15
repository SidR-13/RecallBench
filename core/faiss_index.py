import pickle
import threading

import faiss
import numpy as np


class FaissHNSW:
    """Thin wrapper around FAISS's IndexHNSWFlat, shaped to match our own
    HNSW's public API (insert(node_id, vector), search(query, k, ef=None))
    so benchmarks/harness.py and the benchmark runner can measure both
    index types through identical code, with no per-index-type branching.

    "Flat" means FAISS stores full (uncompressed) vectors at each graph
    node, same as our own HNSW -- the fair comparison is graph-construction
    quality against graph-construction quality, not against a compressed
    (PQ/SQ) FAISS variant that trades additional accuracy for memory in a
    way our implementation was never asked to.
    """

    def __init__(self, dim, M=16, ef_construction=200):
        # faiss.METRIC_L2 is plain (non-squared) L2, not squared L2 like
        # our HNSW/BruteForce -- but since embeddings are L2-normalized,
        # rankings are identical either way (same reasoning as Batch 1's
        # distance metric decision), and recall@k only depends on rank
        # order, not the raw distance values, so this difference never
        # affects any number this project reports.
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = ef_construction
        self._ids = []
        # search() sets efSearch as a side effect on self.index before
        # calling .search() -- a mutable field on the SAME shared object
        # every request goes through (backend/app/main.py loads exactly
        # one FaissHNSW instance into app.state). Confirmed by direct
        # review: FastAPI runs compare.py's sync route handler in a thread
        # pool, so two concurrent /api/compare requests with different ef
        # values race on this field -- one request can silently search
        # with the OTHER request's ef, corrupting results with no
        # exception raised. Same class of bug as core/embed.py's
        # _model_lock fix (Batch 9); this lock is FaissHNSW's equivalent.
        self._search_lock = threading.Lock()

    def __len__(self):
        return len(self._ids)

    def insert(self, node_id, vector):
        # FAISS assigns its own internal sequential index (0, 1, 2, ...) on
        # each add() call, with no concept of an external node_id. We track
        # the mapping ourselves -- self._ids[i] is the real corpus id of
        # whatever FAISS internally calls vector i -- and this only stays
        # correct because insert() is called in strict append order, never
        # out of order or with gaps.
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self.index.add(vector)
        self._ids.append(node_id)

    def memory_bytes(self):
        """Exact byte count of the built index, via FAISS's own
        serialization -- not process RSS (see CLAUDE_CONTEXT.md Batch 6 log
        for why that measurement was dropped for both index types).

        faiss.serialize_index() writes out the complete index -- the full
        stored vectors (this is a Flat index, so it keeps them uncompressed)
        plus the HNSW graph's own bookkeeping -- as a single binary blob.
        Its length in bytes is deterministic for a given built index, and
        directly comparable to HNSW.memory_bytes(): both now count vectors
        + graph structure, nothing more, nothing less.
        """
        return len(faiss.serialize_index(self.index))

    def save(self, index_path, ids_path):
        """Two files, not one: FAISS has its own C++ serialization format
        for the index itself (faiss.write_index), which knows nothing about
        our external node_id -> internal-index mapping (self._ids) -- that
        part is plain Python, so it's pickled separately alongside it.
        """
        faiss.write_index(self.index, str(index_path))
        with open(ids_path, "wb") as f:
            pickle.dump(self._ids, f)

    @classmethod
    def load(cls, index_path, ids_path):
        # Bypasses __init__ deliberately: __init__ always creates a brand
        # new, empty faiss.IndexHNSWFlat -- there's no way to hand it an
        # already-built one. cls.__new__(cls) makes a bare, uninitialized
        # FaissHNSW instance so its two real attributes can be set directly
        # from the serialized files instead.
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(index_path))
        with open(ids_path, "rb") as f:
            obj._ids = pickle.load(f)
        obj._search_lock = threading.Lock()
        return obj

    def search(self, query, k, ef=None):
        """Approximate k-NN, matching our HNSW.search()'s return shape:
        list[(node_id, distance)], ascending by distance, length <= k.

        k <= 0 returns [] without touching FAISS -- verified directly that
        faiss.IndexHNSWFlat.search(q, k=0) (and negative k) raises a raw,
        uncaught C++ AssertionError, unlike BruteForce.search() (Batch 3),
        which already clamps this same case to a clean empty result.
        """
        if k <= 0:
            return []

        with self._search_lock:
            if ef is not None:
                # efSearch is a mutable attribute on the index, not a
                # per-call argument -- FAISS's search-time recall/latency
                # knob, exactly analogous to our own `ef` parameter.
                # Setting it here (instead of once at construction) is
                # what makes the same ef-sweep this project runs against
                # our HNSW possible against FAISS too. Setting it AND
                # calling .search() must happen atomically under
                # self._search_lock (see __init__) -- otherwise a
                # concurrent call can overwrite this value before this
                # call's .search() reads it.
                self.index.hnsw.efSearch = ef

            query = np.asarray(query, dtype=np.float32).reshape(1, -1)
            distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                # FAISS pads with -1 when the index holds fewer than k
                # vectors -- not a real result, must be skipped rather than
                # returned as a fake match.
                continue
            node_id = self._ids[idx]
            # Same normalization as HNSW.search() (Batch 8) / BruteForce.search()
            # (Batch 3): ids inserted from a NumPy array come back as
            # numpy.int64/str_, which isn't directly JSON-serializable.
            if hasattr(node_id, "item"):
                node_id = node_id.item()
            results.append((node_id, float(dist)))
        return results
