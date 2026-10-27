import numpy as np
from nfs_fixed import *

rng = np.random.default_rng(1)
C, B = 30, 5
true_map = rng.normal(size=C)

# 1. band-power reshape: build channel-major importances whose per-channel mean is true_map
imp = np.zeros(C * B)
for ch in range(C):
    imp[ch*B:(ch+1)*B] = true_map[ch] + rng.normal(0, 0.01, B)
blocks = {"band_power": (0, C * B)}
new = importance_to_topography(imp, blocks, C)
old = imp.reshape(B, C).mean(axis=0)          # the old (buggy) code
print("new r with truth:", round(pearson(new, true_map), 3), "| old r with truth:", round(pearson(old, true_map), 3))
assert pearson(new, true_map) > 0.99 and pearson(old, true_map) < 0.8

# 2. GAT map: fully connected graph with self loops, softmax over incoming edges
n = C
src, dst = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
src, dst = src.ravel(), dst.ravel()
score = rng.normal(size=src.size) + 2.0 * (src == 3)   # electrode 3 is attended to more
alpha = np.zeros_like(score)
for i in range(n):
    m = dst == i
    e = np.exp(score[m]); alpha[m] = e / e.sum()
ei = np.vstack([src, dst])
old_map = np.zeros(n); np.add.at(old_map, dst, alpha); old_map /= np.bincount(dst, minlength=n)   # old code
new_map = attention_to_node_topography(ei, alpha, n)
print("old GAT map spread:", old_map.std(), "| new GAT map spread:", round(new_map.std(), 5), "| argmax new:", new_map.argmax())
assert old_map.std() < 1e-12 and new_map.std() > 1e-3 and new_map.argmax() == 3

# 3. subject bootstrap: noisy fold maps around true_map
folds = true_map + rng.normal(0, 1.0, size=(40, C))
print("subject bootstrap:", [round(v, 3) for v in subject_bootstrap_nfs(folds, true_map)])
print("permutation:", [round(v, 4) for v in permutation_p(new, true_map, n_perm=2000)])
print("ALL TESTS PASSED")

# 4. batch attention over several graphs, with PyG-style node offsets
n_graphs = 4
ei_all, a_all = [], []
for k in range(n_graphs):
    ei_all.append(ei + k * n); a_all.append(alpha)
bm = batch_attention_topography(np.hstack(ei_all), np.concatenate(a_all), n)
assert np.allclose(bm, new_map)
print("batch map matches single-graph map")

# 5. reference alignment
chs = [f"E{i}" for i in range(6)]
ref = np.arange(6.0)
shuffled = [chs[i] for i in (3, 0, 5, 1, 4, 2)]
out = align_reference(ref, chs, shuffled)
assert list(out) == [3, 0, 5, 1, 4, 2]
try:
    align_reference(ref, chs, chs + ["E99"]); raise SystemExit("should have raised")
except ValueError:
    pass
print("reference alignment OK")
print("ALL TESTS PASSED (2)")
