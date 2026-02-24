# POPNET with Toy Sample

This repository implements a reproducible pipeline to construct ego-level
multilayer network metrics using toy POPNET sample datasets.

The pipeline builds a multiplex adjacency matrix and computes structural
network indicators suitable for registry-based analyses.

---

## Pipeline Overview

The workflow consists of three scripts:

1. **01_build_clean_network.py**
   - Validates raw node and edge files
   - Creates deterministic ID mapping (0..N-1)
   - Outputs consistent node/edge files

2. **02_build_mln_library.py**
   - Builds a sparse multiplex adjacency matrix
   - Encodes layers as uint64 bitmasks
   - Saves adjacency artifacts

3. **03_compute_metrics.py**
   - Computes ego-level structural metrics:
     - Degree
     - Excess closure
     - Clustering coefficient

---

## Installation

Create a clean virtual environment and install dependencies:

```bash
pip install -r requirements.txt
