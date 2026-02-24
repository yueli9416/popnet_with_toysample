# 03_compute_metrics.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from mlnlib.mln import MultiLayerNetwork

try:
    import igraph as ig
except ImportError:
    ig = None


def sparse_to_igraph(A, directed: bool) -> "ig.Graph":
    coo = A.tocoo()
    edges = list(zip(coo.row.tolist(), coo.col.tolist()))
    g = ig.Graph(n=A.shape[0], edges=edges, directed=directed)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--lib_dir", type=str, default="toy_library_manual")
    ap.add_argument("--layers_file", type=str, default="data/layer_toy_FIXED.csv")
    ap.add_argument("--out_dir", type=str, default="metrics")

    # Core vs optional heavy metrics
    ap.add_argument("--compute_centrality", action="store_true",
                    help="Compute betweenness/closeness/eigenvector via igraph (NOT feasible for full POPNET).")
    ap.add_argument("--centrality_sample_n", type=int, default=0,
                    help="If >0, compute centralities on a random sample of nodes of this size.")
    ap.add_argument("--undirected", action="store_true",
                    help="Treat graph as undirected for igraph centralities (use if you symmetrized A in pipeline 02).")
    ap.add_argument("--giant_component_only", action="store_true",
                    help="Compute centralities only on the giant component (recommended).")
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()

    root = Path(args.root)
    lib = root / args.lib_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    A_path = lib / "A_multiplex.npz"
    nodes_path = lib / "nodes_for_mlnlib.csv"
    layers_path = root / args.layers_file

    if not A_path.exists():
        raise FileNotFoundError(A_path)
    if not nodes_path.exists():
        raise FileNotFoundError(nodes_path)
    if not layers_path.exists():
        raise FileNotFoundError(layers_path)

    A = load_npz(A_path).tocsr()
    nodes_df = pd.read_csv(nodes_path)
    layers_df = pd.read_csv(layers_path)

    # Basic checks
    N = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square. Got {A.shape}")
    if nodes_df["id"].nunique() != N:
        raise ValueError(f"nodes_for_mlnlib has {nodes_df['id'].nunique()} ids but A has N={N}")

    # Build MLN (core metrics)
    mln = MultiLayerNetwork(edges=A, nodes=nodes_df, layers=layers_df, verbose=False)

    deg = mln.get_degrees()
    excl_all = mln.get_excess_closure()
    clustering = excl_all["clustering_coefficient"]
    excess_closure = excl_all["excess_closure"]

    core = pd.DataFrame({"id": np.arange(N, dtype=int)})
    core["degree"] = core["id"].map(deg)
    core["clustering"] = core["id"].map(clustering)
    core["excess_closure"] = core["id"].map(excess_closure)

    core = core.merge(nodes_df[["id", "label"]], on="id", how="left").sort_values("id")

    core_path = out_dir / "node_metrics_core.csv"
    core.to_csv(core_path, index=False)

    # Metadata
    meta = {
        "N": int(N),
        "nnz_directed_ties": int(A.nnz),
        "core_metrics": ["degree", "clustering", "excess_closure"],
        "centrality_computed": bool(args.compute_centrality),
        "centrality_sample_n": int(args.centrality_sample_n),
        "undirected_for_centrality": bool(args.undirected),
        "giant_component_only": bool(args.giant_component_only),
        "note": "Centralities are optional and generally not feasible for full POPNET; use sampling/GC."
    }

    # Optional centralities
    if args.compute_centrality:
        if ig is None:
            raise ImportError("igraph not installed. Install python-igraph or disable --compute_centrality.")

        rng = np.random.default_rng(args.seed)
        keep = np.arange(N)

        if args.centrality_sample_n and args.centrality_sample_n > 0:
            if args.centrality_sample_n >= N:
                keep = np.arange(N)
            else:
                keep = np.sort(rng.choice(np.arange(N), size=args.centrality_sample_n, replace=False))

        # Subgraph adjacency
        A_sub = A[keep, :][:, keep]
        directed = not args.undirected

        g = sparse_to_igraph(A_sub, directed=directed)

        # Giant component restriction
        if args.giant_component_only:
            comps = g.clusters(mode="WEAK" if directed else "STRONG")
            gc_idx = int(np.argmax(comps.sizes()))
            gc_vertices = comps.subgraph(gc_idx).vs.indices  # indices in the subgraph
            g_gc = g.subgraph(gc_vertices)
            meta["centrality_component"] = "giant_component"
            meta["centrality_component_size"] = int(g_gc.vcount())
            # compute on GC only
            bet = g_gc.betweenness(directed=directed)
            clo = g_gc.closeness(mode="OUT" if directed else "ALL")
            eig = g_gc.eigenvector_centrality(directed=directed, scale=True)

            # map back to full N with NaNs
            out = core.copy()
            out["betweenness"] = np.nan
            out["closeness"] = np.nan
            out["eigenvector"] = np.nan

            # vertices in GC correspond to positions in keep[gc_vertices]
            ids_gc = keep[np.array(gc_vertices, dtype=int)]
            out.loc[out["id"].isin(ids_gc), "betweenness"] = bet
            out.loc[out["id"].isin(ids_gc), "closeness"] = clo
            out.loc[out["id"].isin(ids_gc), "eigenvector"] = eig

        else:
            meta["centrality_component"] = "all_nodes_in_subgraph"
            meta["centrality_component_size"] = int(g.vcount())

            bet = g.betweenness(directed=directed)
            clo = g.closeness(mode="OUT" if directed else "ALL")
            eig = g.eigenvector_centrality(directed=directed, scale=True)

            out = core.copy()
            out["betweenness"] = np.nan
            out["closeness"] = np.nan
            out["eigenvector"] = np.nan
            out.loc[out["id"].isin(keep), "betweenness"] = bet
            out.loc[out["id"].isin(keep), "closeness"] = clo
            out.loc[out["id"].isin(keep), "eigenvector"] = eig

        out_path = out_dir / "node_metrics_with_centrality.csv"
        out.to_csv(out_path, index=False)

        print("Saved:", out_path)

    meta_path = out_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Saved:", core_path)
    print("Saved:", meta_path)
    print("Mean degree:", float(core["degree"].mean()))
    print("Mean excess_closure:", float(core["excess_closure"].mean()))


if __name__ == "__main__":
    main()