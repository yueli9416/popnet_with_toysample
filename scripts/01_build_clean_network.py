# 01_build_clean_network.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def build_id_map(nodes: pd.DataFrame, label_col: str) -> dict:
    labels = nodes[label_col].tolist()
    return {lab: i for i, lab in enumerate(labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=r"your_path\toy_run",
        help="Project root"
    )
    ap.add_argument("--nodes", type=str, default="data/popnet_nodelist.csv")
    ap.add_argument("--edges", type=str, default="data/popnet_edgelist.csv")
    ap.add_argument("--layers", type=str, default="data/layer_toy_FIXED.csv")
    ap.add_argument("--node_label_col", type=str, default="label")
    ap.add_argument("--edge_source_col", type=str, default="source")
    ap.add_argument("--edge_target_col", type=str, default="target")
    ap.add_argument("--edge_layer_col", type=str, default="layer")
    ap.add_argument("--out_nodes", type=str, default="data/popnet_nodelist_consistent.csv")
    ap.add_argument("--out_edges", type=str, default="data/popnet_edgelist_consistent.csv")
    args = ap.parse_args()

    root = Path(args.root)
    nodes_path = root / args.nodes
    edges_path = root / args.edges
    layers_path = root / args.layers

    if not nodes_path.exists():
        raise FileNotFoundError(f"Nodes not found: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Edges not found: {edges_path}")
    if not layers_path.exists():
        raise FileNotFoundError(f"Layers not found: {layers_path}")

    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    layers = pd.read_csv(layers_path)

    # --- basic schema checks ---
    for c in [args.node_label_col]:
        if c not in nodes.columns:
            raise ValueError(f"nodes missing column '{c}'. Found: {list(nodes.columns)}")
    for c in [args.edge_source_col, args.edge_target_col, args.edge_layer_col]:
        if c not in edges.columns:
            raise ValueError(f"edges missing column '{c}'. Found: {list(edges.columns)}")
    if "layer" not in layers.columns:
        raise ValueError(f"layers missing 'layer'. Found: {list(layers.columns)}")

    # --- layer compatibility ---
    edge_layers = set(edges[args.edge_layer_col].astype(str).unique())
    layer_layers = set(layers["layer"].astype(str).unique())
    extra_in_edges = edge_layers - layer_layers
    extra_in_layers = layer_layers - edge_layers
    if extra_in_edges:
        raise ValueError(f"Layers in edges but not layers file: {extra_in_edges}")
    if extra_in_layers:
        print(f"WARNING: Layers in layers file but not edges (ok): {sorted(extra_in_layers)[:10]}")

    # --- build mapping label -> 0..N-1 based on nodes order ---
    # Keep node order stable
    nodes = nodes.copy()
    nodes[args.node_label_col] = nodes[args.node_label_col].astype(int)
    nodes = nodes.drop_duplicates(subset=[args.node_label_col]).reset_index(drop=True)

    id_map = build_id_map(nodes, args.node_label_col)

    # --- remap edges ---
    edges = edges.copy()
    edges[args.edge_source_col] = edges[args.edge_source_col].astype(int)
    edges[args.edge_target_col] = edges[args.edge_target_col].astype(int)

    # verify edges labels exist in nodes
    edge_labels = set(pd.concat([edges[args.edge_source_col], edges[args.edge_target_col]]).unique())
    node_labels = set(nodes[args.node_label_col].unique())
    missing = edge_labels - node_labels
    if missing:
        raise ValueError(f"Edges reference labels not in nodes. Missing count={len(missing)} sample={list(sorted(missing))[:10]}")

    edges_mapped = edges.copy()
    edges_mapped["source"] = edges_mapped[args.edge_source_col].map(id_map)
    edges_mapped["target"] = edges_mapped[args.edge_target_col].map(id_map)
    edges_mapped["layer"] = edges_mapped[args.edge_layer_col].astype(str)


    N = len(nodes)
    max_endpoint = int(max(edges_mapped["source"].max(), edges_mapped["target"].max()))
    if max_endpoint >= N:
        raise ValueError(f"Mapping failed: max endpoint {max_endpoint} >= N {N}")

    # output node list: id,label
    out_nodes = pd.DataFrame({"id": range(N), "label": nodes[args.node_label_col].tolist()})
    out_edges = edges_mapped[["source", "target", "layer"]].copy()

    out_nodes_path = root / args.out_nodes
    out_edges_path = root / args.out_edges
    out_nodes_path.parent.mkdir(parents=True, exist_ok=True)

    out_nodes.to_csv(out_nodes_path, index=False)
    out_edges.to_csv(out_edges_path, index=False)

    print(f"N nodes: {N}")
    print(f"Edges rows: {len(out_edges)} | unique layers: {out_edges['layer'].nunique()}")
    print(f"Wrote nodes: {out_nodes_path}")
    print(f"Wrote edges: {out_edges_path}")


if __name__ == "__main__":
    main()