# 02_build_mln_library.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--nodes", type=str, default="data/popnet_nodelist_consistent.csv")
    ap.add_argument("--edges", type=str, default="data/popnet_edgelist_consistent.csv")
    ap.add_argument("--layers", type=str, default="data/layer_toy_FIXED.csv")
    ap.add_argument("--out_dir", type=str, default="toy_library_manual")

    ap.add_argument("--symmetrize", action="store_true", help="Make network undirected using bitwise OR with transpose")
    ap.add_argument("--drop_self_loops", action="store_true", help="Drop edges where source==target")

    args = ap.parse_args()

    root = Path(args.root)
    nodes_path = root / args.nodes
    edges_path = root / args.edges
    layers_path = root / args.layers
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load ---
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    layers = pd.read_csv(layers_path)

    # --- validate schema ---
    for c in ["id", "label"]:
        if c not in nodes.columns:
            raise ValueError(f"nodes missing '{c}', found {list(nodes.columns)}")
    for c in ["source", "target", "layer"]:
        if c not in edges.columns:
            raise ValueError(f"edges missing '{c}', found {list(edges.columns)}")
    for c in ["layer", "binary"]:
        if c not in layers.columns:
            raise ValueError(f"layers missing '{c}', found {list(layers.columns)}")

    # --- validate node ids ---
    N = int(nodes["id"].nunique())
    if nodes["id"].min() != 0 or nodes["id"].max() != N - 1:
        raise ValueError(f"Node ids must be contiguous 0..N-1. Got min={nodes['id'].min()} max={nodes['id'].max()} N={N}")

    # --- validate layers mapping ---
    layers["layer"] = layers["layer"].astype(str)
    edges["layer"] = edges["layer"].astype(str)

    layer_to_bit = dict(zip(layers["layer"], layers["binary"].astype(np.uint64)))

    extra = sorted(set(edges["layer"]) - set(layer_to_bit.keys()))
    if extra:
        raise ValueError(f"Edges contain layers not found in layers file: {extra[:10]} (showing first 10)")

    # --- clean edges ---
    edges["source"] = edges["source"].astype(np.int64)
    edges["target"] = edges["target"].astype(np.int64)

    if args.drop_self_loops:
        before = len(edges)
        edges = edges[edges["source"] != edges["target"]].copy()
        print(f"Dropped self-loops: {before - len(edges)}")

    # --- validate endpoints ---
    if edges["source"].min() < 0 or edges["target"].min() < 0:
        raise ValueError("Found negative node ids in edges.")
    if edges["source"].max() >= N or edges["target"].max() >= N:
        raise ValueError(f"Found node id >= N in edges. max_source={edges['source'].max()} max_target={edges['target'].max()} N={N}")

    # --- diagnostics: edges per layer ---
    layer_counts = edges["layer"].value_counts()
    print("Edges per layer (top 10):")
    print(layer_counts.head(10).to_string())

    # --- encode layers as bitmasks ---
    edges["bit"] = edges["layer"].map(layer_to_bit).astype(np.uint64)

    # --- aggregate duplicates ---
    grouped = edges.groupby(["source", "target"], as_index=False)["bit"].agg(lambda x: np.bitwise_or.reduce(x.to_numpy(dtype=np.uint64)))
    src = grouped["source"].to_numpy()
    tgt = grouped["target"].to_numpy()
    bits = grouped["bit"].to_numpy(dtype=np.uint64)

    # --- build sparse multiplex adjacency ---
    A = csr_matrix((bits, (src, tgt)), shape=(N, N), dtype=np.uint64)
    A.sum_duplicates()

    # --- optional symmetrization ---
    if args.symmetrize:
        AT = A.transpose().tocsr()
        A = A.maximum(AT)
        print("Applied symmetrization (A = max(A, A^T)).")

    # --- save artifacts ---
    save_npz(out_dir / "A_multiplex.npz", A)
    nodes[["id", "label"]].to_csv(out_dir / "nodes_for_mlnlib.csv", index=False)
    layers.to_csv(out_dir / "layers_for_mlnlib.csv", index=False)

    print(f"N = {N}")
    print(f"Dyads (unique source-target pairs) = {len(grouped)}")
    print(f"Multiplex nnz (directed ties) = {A.nnz} | max bitmask = {int(A.data.max()) if A.nnz else 0}")
    print(f"Saved: {out_dir / 'A_multiplex.npz'}")
    print(f"Saved: {out_dir / 'nodes_for_mlnlib.csv'}")
    print(f"Saved: {out_dir / 'layers_for_mlnlib.csv'}")


if __name__ == "__main__":
    main()