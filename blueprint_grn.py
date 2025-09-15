# v0.1 Blueprint GRN — executable prototype
# Builds a simple developmental blueprint graph (iris/retina/cortex),
# simulates genotype effects, propagates through the network,
# and computes a DNA⇄Iris consistency score.
#
# Notes for this demo:
# - Uses iterative propagation on a directed weighted graph.
# - Predicts 5 iris features from module activities via a linear head.
# - Compares to a dummy observed iris feature vector.
#
# Authors: ChatGPT and Jacob Thomas Messer
#
# You can replace the simulated genotype and observed features with real data later.

import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv

# -------------------------------
# 1) Define nodes (modules/genes)
# -------------------------------
nodes = [
    # Global lineage / timing
    "PAX6", "PAX3", "SOX10", "NEUROECT", "NEURAL_CREST", "TEMPORAL_SWITCH_Notch",
    # Iris modules
    "MITF", "TYR", "TYRP1", "DCT", "OCA2", "HERC2", "SLC24A5", "SLC45A2",
    "IRIS_PIGMENT", "IRIS_STROMA", "IRIS_MUSCLE", "IRIS_VASC_IMM", "IRIS_CRyPTS_FOLDS",
    # Retina modules
    "RPC_CORE", "RET_NEUROGEN", "RET_GLIA",
    # Cortex modules
    "RG_CORE", "INTERMEDIATE_PROG", "CORTEX_LAYERS"
]

idx = {n:i for i,n in enumerate(nodes)}

# -------------------------------
# 2) Build graph with weighted edges
# -------------------------------
G = nx.DiGraph()
G.add_nodes_from(nodes)

def add(src, tgt, w):
    G.add_edge(src, tgt, weight=w)

# Core developmental relationships
add("PAX6", "NEUROECT", 0.9)
add("PAX3", "NEURAL_CREST", 0.9)
add("SOX10", "NEURAL_CREST", 0.8)

# Iris differentiation
add("NEUROECT", "IRIS_MUSCLE", 0.6)
add("NEURAL_CREST", "IRIS_STROMA", 0.7)
add("NEURAL_CREST", "IRIS_VASC_IMM", 0.6)

# Pigment regulation
add("MITF", "TYR", 0.85)
add("MITF", "TYRP1", 0.8)
add("MITF", "DCT", 0.75)
add("HERC2", "OCA2", 0.8)
add("OCA2", "IRIS_PIGMENT", 0.6)
add("SLC24A5", "IRIS_PIGMENT", 0.4)
add("SLC45A2", "IRIS_PIGMENT", 0.4)

# Iris visible features
add("IRIS_STROMA", "IRIS_CRyPTS_FOLDS", 0.5)
add("IRIS_MUSCLE", "IRIS_CRyPTS_FOLDS", 0.5)
add("IRIS_PIGMENT", "IRIS_CRyPTS_FOLDS", 0.2)  # pigment can influence mechanics subtly
add("IRIS_PIGMENT", "IRIS_STROMA", 0.1)        # reciprocal soft coupling

# Retina
add("RPC_CORE", "RET_NEUROGEN", 0.7)
add("RPC_CORE", "RET_GLIA", 0.5)

# Cortex
add("RG_CORE", "INTERMEDIATE_PROG", 0.7)
add("INTERMEDIATE_PROG", "CORTEX_LAYERS", 0.7)

# Temporal switch (Notch) inhibitions
add("TEMPORAL_SWITCH_Notch", "RET_NEUROGEN", -0.5)
add("TEMPORAL_SWITCH_Notch", "CORTEX_LAYERS", -0.5)

# -------------------------------
# 3) Adjacency matrix (column-normalized for stability)
# -------------------------------
n = len(nodes)
A = np.zeros((n,n))
for u,v,data in G.edges(data=True):
    A[idx[u], idx[v]] = data["weight"]

# Normalize columns to avoid blow-up in propagation
col_sums = np.maximum(np.sum(np.abs(A), axis=0), 1e-9)
A_norm = A / col_sums

# -------------------------------
# 4) Simulate genotype effects (g) and observed features (replace with real data later)
# -------------------------------
g = np.zeros(n)
# Simulate variants that impact pigment regulation
g[idx["HERC2"]]  = 1.0   # regulatory variant
g[idx["OCA2"]]   = 0.5
g[idx["MITF"]]   = 0.3
g[idx["PAX6"]]   = 0.2
g[idx["SOX10"]]  = 0.1

g = g / (norm(g) + 1e-9)  # normalize

# Observed iris feature vector (dummy): [pigment_intensity, eumel:pheo, pH_proxy, stromal_scatter, fold_density]
iris_obs = np.array([0.8, 0.4, 0.3, 0.6, 0.7])

# Retina and cortex (minimal use here; could be expanded for cross-tissue penalties)
retina_obs = np.array([0.6, 0.3, 0.9])   # [photoreceptor_fraction, glia_fraction, maturity]
cortex_obs = np.array([0.8, 0.7, 0.5])   # [RG→IP flux, deep:upper ratio, gliogenesis]

# -------------------------------
# 5) Propagate genotype through network to get module activities
# x_{t+1} = tanh(A_norm^T x_t + g_bias)
# -------------------------------
def propagate(A_norm, g, steps=12):
    x = np.copy(g)  # start with genotype prior as bias
    for _ in range(steps):
        x = np.tanh(A_norm.T @ x + g)
    return x

x_modules = propagate(A_norm, g, steps=16)

# -------------------------------
# 6) Map module activities to 5 iris features via a fixed linear head (v0.1)
# You can replace these weights with learned values later.
# -------------------------------
Phi = np.zeros((5, n))  # 5 features × n modules

# Feature 0: pigment_intensity ← IRIS_PIGMENT (+), TYR/TYRP1 (+), OCA2/HERC2 (+)
Phi[0, idx["IRIS_PIGMENT"]] = 0.6
Phi[0, idx["TYR"]]          = 0.1
Phi[0, idx["TYRP1"]]        = 0.1
Phi[0, idx["OCA2"]]         = 0.1
Phi[0, idx["HERC2"]]        = 0.1

# Feature 1: eumelanin:pheomelanin proxy ← TYR(+), DCT(+), SLC24A5(+), SLC45A2(+)
Phi[1, idx["TYR"]]      = 0.35
Phi[1, idx["DCT"]]      = 0.25
Phi[1, idx["SLC24A5"]]  = 0.20
Phi[1, idx["SLC45A2"]]  = 0.20

# Feature 2: melanosome pH proxy ← SLC24A5(+), SLC45A2(+), OCA2(+)
Phi[2, idx["SLC24A5"]]  = 0.4
Phi[2, idx["SLC45A2"]]  = 0.3
Phi[2, idx["OCA2"]]     = 0.3

# Feature 3: stromal_scatter ← IRIS_STROMA(+), IRIS_VASC_IMM(+)
Phi[3, idx["IRIS_STROMA"]]  = 0.7
Phi[3, idx["IRIS_VASC_IMM"]] = 0.3

# Feature 4: fold_density ← IRIS_CRyPTS_FOLDS(+), IRIS_MUSCLE(+)
Phi[4, idx["IRIS_CRyPTS_FOLDS"]] = 0.6
Phi[4, idx["IRIS_MUSCLE"]]       = 0.4

pred_iris = Phi @ x_modules

# Clip to [0,1] for interpretability
pred_iris = np.clip((pred_iris - pred_iris.min()) / (pred_iris.max() - pred_iris.min() + 1e-9), 0, 1)

# -------------------------------
# 7) Consistency score (negative squared error + simple cross-penalty)
# -------------------------------
err = norm(pred_iris - iris_obs)
cross_penalty = 0.5 * abs(retina_obs[-1] - pred_iris[0])  # maturity vs pigment intensity
score = - (err + cross_penalty)

# -------------------------------
# 8) Visualize the graph (one plot, default colors per instructions)
# -------------------------------
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=7)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos, arrows=True)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Developmental Blueprint GRN (v0.1)")
plt.axis('off')
plt.show()

# -------------------------------
# 9) Save artifacts for you to download / reuse
# -------------------------------
artifact = {
    "nodes": nodes,
    "edges": [{"source": u, "target": v, "weight": float(d["weight"])} for u,v,d in G.edges(data=True)],
    "adjacency_column_normalized": A_norm.tolist(),
    "genotype_vector": g.tolist(),
    "module_activity": x_modules.tolist(),
    "predicted_iris_features": pred_iris.tolist(),
    "observed_iris_features": iris_obs.tolist(),
    "score": float(score)
}

path = "/mnt/data/blueprint_grn_v0_1.json"
with open(path, "w") as f:
    json.dump(artifact, f, indent=2)

print("DNA⇄Iris Consistency Score:", score)
print("Predicted Iris Feature Vector:", pred_iris)
print("Observed Iris Feature Vector:", iris_obs)
print(f"\nSaved model artifact: {path}")
