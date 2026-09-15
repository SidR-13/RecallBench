import heapq
import math
import pickle
import random
import sys

import numpy as np


class HNSW:
    def __init__(self, dim, M=16, ef_construction=200, seed=42):
        self.dim = dim
        self.M = M
        self.M0 = 2 * M  # layer 0 gets double the degree budget: it carries all
                          # fine-grained connectivity, every other layer only needs
                          # to provide coarse long-range shortcuts
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)  # normalizes the level distribution so each
                                      # layer is expected to be ~1/M the size of
                                      # the one below it

        self.vectors = {}
        self.graph = []  # graph[layer] -> {node_id: set(neighbor_ids)}
        self.entry_point = None
        self.max_level = -1

        # Private RNG instance (not the global `random` module) so index
        # construction stays reproducible regardless of what else in the
        # process calls random.* — needed so benchmark runs are repeatable.
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.vectors)

    def memory_bytes(self):
        """Exact byte count of everything this index actually owns: the
        vector copies in self.vectors plus the graph structure in
        self.graph (per-layer dicts of node_id -> set(neighbor_ids)).

        Deliberately NOT process-wide RSS (resource.getrusage) -- measured
        directly (see CLAUDE_CONTEXT.md Batch 6 log) that ru_maxrss swings
        by 100s of MB run-to-run for the identical build, driven by
        allocator/heap-layout noise from whatever else the process touched
        (corpus loading, ground-truth computation) rather than the graph's
        own footprint. Walking the actual live objects is deterministic:
        the same graph always reports the same number, on any machine.

        sys.getsizeof(obj) alone would undercount container types -- it
        reports only the container's own overhead, not what it holds (e.g.
        a dict's reported size doesn't include its keys/values). So this
        recurses into dicts/sets/lists explicitly. It does NOT recurse into
        numpy arrays' contents: sys.getsizeof on an ndarray already
        includes the full data buffer (unlike a Python container), so
        recursing further would double-count.

        `seen` deduplicates by id() (object identity, not value) so a
        node_id object referenced from multiple places -- e.g. the same id
        appearing as a dict key in self.vectors, a dict key in a graph
        layer, AND inside a neighbor set -- has its own memory counted once,
        not N times. Small ints are cached/interned by CPython so this also
        naturally avoids inflating counts for integer ids.
        """
        seen = set()

        def deep_size(obj):
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            seen.add(obj_id)

            size = sys.getsizeof(obj)
            if isinstance(obj, dict):
                for key, value in obj.items():
                    size += deep_size(key)
                    size += deep_size(value)
            elif isinstance(obj, (set, frozenset, list, tuple)):
                for item in obj:
                    size += deep_size(item)
            return size

        return deep_size(self.vectors) + deep_size(self.graph)

    def save(self, path):
        """Pickle the whole index -- every attribute (self.vectors,
        self.graph, self._rng, plain ints/floats) is already ordinary,
        safely-picklable Python/NumPy data, so there's no custom
        serialization format to design here.

        Exists so a real API server (Batch 8) can build the index once, via
        a script, and load it back in a few seconds on startup -- a fresh
        100k-vector build takes ~3.5 minutes (Batch 6), far too slow to
        repeat every time the server restarts.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def _distance(self, vec_id_a, vec_id_b):
        # Squared L2, not plain L2: sqrt is monotonic and wasted work for
        # ranking. Once embeddings are L2-normalized (Batch 4), squared-L2
        # ranking is mathematically identical to cosine ranking.
        diff = self.vectors[vec_id_a] - self.vectors[vec_id_b]
        return float(np.dot(diff, diff))

    def _distance_to_query(self, query, vec_id):
        # Same as _distance, but for a query vector that isn't (yet) stored
        # in self.vectors — used during both insertion and search.
        diff = query - self.vectors[vec_id]
        return float(np.dot(diff, diff))

    def _random_level(self):
        # Inverse-CDF sampling from an exponential distribution: turns a
        # uniform draw into the exponentially-decaying level distribution
        # HNSW depends on for its logarithmic search shortcuts. r can in
        # theory be exactly 0.0 (log(0) is undefined), so re-draw if it is.
        r = 0.0
        while r == 0.0:
            r = self._rng.random()
        return int(-math.log(r) * self.mL)

    def _search_layer(self, query, entry_points, ef, layer):
        """Bounded best-first beam search within a single layer.

        Not the public query API (that's Batch 2) — this is the primitive
        insertion uses internally to find candidate neighbors for a new node.
        """
        visited = set(entry_points)
        candidates = [(self._distance_to_query(query, ep), ep) for ep in entry_points]
        heapq.heapify(candidates)  # min-heap: pop the closest unexplored node next

        # `found` holds the current best `ef` results. heapq is a min-heap,
        # but we need cheap access to the WORST kept result (to know when to
        # evict it), so distances are negated — the heap root then holds the
        # most negative value, i.e. the largest real distance.
        found = [(-d, ep) for d, ep in candidates]
        heapq.heapify(found)

        while candidates:
            cur_dist, cur_id = heapq.heappop(candidates)
            worst_found_dist = -found[0][0]
            # Early termination: if even the closest remaining candidate is
            # worse than our current worst kept result, nothing left to
            # explore can improve `found`. This is a heuristic bound (not a
            # proof) — ef_construction is set generously to compensate.
            if cur_dist > worst_found_dist and len(found) >= ef:
                break

            for neighbor_id in self.graph[layer].get(cur_id, ()):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                d = self._distance_to_query(query, neighbor_id)
                worst_found_dist = -found[0][0]
                if len(found) < ef or d < worst_found_dist:
                    heapq.heappush(candidates, (d, neighbor_id))
                    heapq.heappush(found, (-d, neighbor_id))
                    if len(found) > ef:
                        heapq.heappop(found)  # evict current worst, keeps found <= ef

        return [(-neg_d, node_id) for neg_d, node_id in found]  # un-negate distances

    def _greedy_descend(self, query, ep, from_layer, to_layer):
        """Cheap ef=1 greedy walk down through layers, narrowing the entry
        point before real (wide-beam) work happens at `to_layer`. Shared by
        insert() (descending to the new node's own level) and search()
        (always descending to layer 0).
        """
        for layer in range(from_layer, to_layer, -1):
            nearest = self._search_layer(query, ep, ef=1, layer=layer)
            ep = [nearest[0][1]]
        return ep

    def _select_neighbors(self, candidates, m):
        # Naive top-M nearest. The HNSW paper's diversity-aware heuristic
        # (avoiding neighbors clustered in one direction) is deliberately
        # deferred — see CLAUDE_CONTEXT.md Batch 1 log for why.
        return sorted(candidates, key=lambda pair: pair[0])[:m]

    def _ensure_layer_exists(self, layer):
        # Layers are created lazily, only once some node's random level
        # actually reaches them — no need to pre-guess the max layer count.
        while len(self.graph) <= layer:
            self.graph.append({})

    def _add_edge(self, layer, a, b):
        # Edges are always bidirectional: HNSW's navigability guarantee
        # requires that if search can walk a -> b, it can also walk b -> a.
        # A one-way edge would silently create unreachable regions.
        self.graph[layer].setdefault(a, set()).add(b)
        self.graph[layer].setdefault(b, set()).add(a)

    def _prune_connections(self, layer, node_id, max_degree):
        # Enforces the per-layer degree cap. Without this, popular "hub"
        # nodes would accumulate unbounded neighbors over time, search would
        # degrade toward brute force, and memory would grow without bound.
        neighbor_ids = self.graph[layer][node_id]
        if len(neighbor_ids) <= max_degree:
            return

        candidates = [(self._distance(node_id, nid), nid) for nid in neighbor_ids]
        kept = self._select_neighbors(candidates, max_degree)
        kept_ids = {nid for _, nid in kept}

        # Must remove the reverse edge too — edges are bidirectional, so
        # dropping only one side would leave a dangling one-way edge.
        for nid in neighbor_ids - kept_ids:
            self.graph[layer][nid].discard(node_id)

        self.graph[layer][node_id] = kept_ids

    def insert(self, node_id, vector):
        # copy=True (not np.asarray) forces the index to own its data,
        # rather than aliasing whatever array the caller passed in. Without
        # this, np.asarray only copies when strictly necessary (wrong
        # dtype/type) -- given our callers already pass float32 rows, it
        # would silently alias the caller's array: a correctness risk if
        # the caller ever mutates/frees it later, and it also meant the
        # index's own memory footprint (Batch 5's benchmark) excluded
        # vector storage entirely, understating what a real standalone
        # deployed index actually costs.
        vector = np.array(vector, dtype=np.float32, copy=True)
        if vector.shape != (self.dim,):
            # self.dim was previously stored but never checked against
            # anything (found during an exhaustive review) -- a
            # wrong-dimension vector was accepted silently, with the
            # failure only surfacing later, deep inside _distance's numpy
            # broadcast, as a confusing error unrelated to its real cause
            # (or not erroring at all, if shapes happened to broadcast).
            # FaissHNSW gets an equivalent check for free from FAISS's C
            # API; this makes both index types fail the same way.
            raise ValueError(
                f"expected vector of shape ({self.dim},), got {vector.shape}"
            )
        self.vectors[node_id] = vector
        level = self._random_level()

        if self.entry_point is None:
            # First node in the index: nothing to search or connect to yet.
            # Just register it (neighborless) at every layer up to its level.
            self._ensure_layer_exists(level)
            for layer in range(level + 1):
                self.graph[layer][node_id] = set()
            self.entry_point = node_id
            self.max_level = level
            return

        # Phase 1: descend through layers above this node's own level.
        # The new node doesn't exist up here, so no edges are built — this
        # just narrows down a good entry point via cheap greedy (ef=1) walks
        # before we reach the layer where real wiring starts.
        ep = self._greedy_descend(vector, [self.entry_point], self.max_level, level)

        # Phase 2: from min(max_level, level) down to layer 0, actually wire
        # the new node in. min(...) matters when this node's level exceeds
        # the index's current max_level — there's no existing graph above
        # the old max_level to search, so wiring can only start there.
        self._ensure_layer_exists(level)
        for layer in range(min(self.max_level, level), -1, -1):
            self.graph[layer].setdefault(node_id, set())

            # Wide beam search (ef_construction) — this is where connection
            # quality is actually decided, unlike the ef=1 walk in phase 1.
            candidates = self._search_layer(vector, ep, self.ef_construction, layer)
            max_degree = self.M0 if layer == 0 else self.M
            neighbors = self._select_neighbors(candidates, max_degree)

            for _, neighbor_id in neighbors:
                self._add_edge(layer, node_id, neighbor_id)

            # Adding an edge may have pushed a neighbor over its degree cap.
            for _, neighbor_id in neighbors:
                self._prune_connections(layer, neighbor_id, max_degree)

            # Entry points for the next (lower) layer's search: use the full
            # candidate list found here, not just the selected neighbors —
            # richer entry points improve the next layer's search quality.
            ep = [nid for _, nid in candidates]

        if level > self.max_level:
            self.entry_point = node_id
            self.max_level = level

    def search(self, query, k, ef=None):
        """Approximate k-nearest-neighbor query.

        `ef` is the layer-0 beam width — the recall/latency knob this whole
        project benchmarks. Larger ef explores more candidates: higher
        recall, higher latency. Defaults to max(k, 50) when not given; the
        real recall/latency tradeoff at various ef values is measured for
        real in Batch 5, not assumed here.

        Returns list[(node_id, distance)], ascending by distance, length
        <= k. Distances are squared L2 (see _distance), not true Euclidean
        — ranking is identical either way and sqrt would be wasted work.
        node_id is converted back to its native Python type (matching
        BruteForce.search()'s Batch 3 precedent) since ids inserted from a
        NumPy array (e.g. the real corpus) are numpy.int64/str_, which
        Batch 8's FastAPI layer cannot JSON-serialize directly.

        k <= 0 returns [] -- without this, `candidates[:k]` below would
        silently slice from the END for negative k (Python slice
        semantics), returning len(candidates)+k wrong results instead of
        erroring or returning none. Verified directly: k=-1 against a
        20-node test index returned 19 (nearly all) results. Matches
        BruteForce.search()'s existing k<=0 handling (Batch 3).
        """
        if k <= 0:
            return []

        if self.entry_point is None:
            return []

        query = np.asarray(query, dtype=np.float32)
        if ef is None:
            ef = max(k, 50)
        ef = max(ef, 1)  # a non-positive ef would make _search_layer's
        # `len(found) >= ef` check always true, terminating the beam
        # search almost immediately -- silently degrading search quality
        # rather than erroring. Flooring at 1 keeps behavior sane for any
        # caller-supplied ef, matching k's own floor above.

        # Same cheap greedy descent insert() uses, all the way to layer 0.
        ep = self._greedy_descend(query, [self.entry_point], self.max_level, 0)

        # Wide beam search at layer 0, where every node lives.
        candidates = self._search_layer(query, ep, max(ef, k), layer=0)
        candidates.sort(key=lambda pair: pair[0])

        return [
            (node_id.item() if hasattr(node_id, "item") else node_id, dist)
            for dist, node_id in candidates[:k]
        ]
