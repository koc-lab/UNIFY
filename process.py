"""Data loading and preprocessing for UNIFY.

================================================================================
                           WHERE TO PUT THE DATA
================================================================================
No data ships with this repository. Everything is read from a single root folder
(``--data_dir``, default ``../Data`` relative to this file). Create it yourself
and populate it exactly as described below.

--------------------------------------------------------------------------------
SPATIAL DOMAIN IDENTIFICATION DATA (DLPFC, 12 sections)
--------------------------------------------------------------------------------
Data/
  1-DLPFC/<section>/                                  # <section> in DLPFC_SECTIONS
      filtered_feature_bc_matrix.h5                   # 10x Visium RAW counts (CellRanger
                                                      #   output, NOT normalized).
      spatial/                                        # The ORIGIN of the spot coordinates
          tissue_positions_list.txt                   #   UNIFY trains on, though they reach
            (or tissue_positions_list.csv,            #   it via the backbone's
             tissue_positions.csv,                    #   <section>_spatial.npy (see below).
             tissue_positions.parquet)                #   ANY of these four filenames is
          scalefactors_json.json                      #   accepted, - see
                                                      #   read_tissue_positions.                                                                                                                                                 
          <section>_A_coexpr_k<k>.npz                 # co-expression adjacency cache, one
          <section>_A_coexpr_k<k>_barcodes.npy        #   pair per k. OPTIONAL: a run loads
                                                      #   them when present and generates +
                                                      #   saves them here when absent, so a
                                                      #   fresh Data/ fills itself on the
                                                      #   first run.
      image_representation/
          phikon_representation.csv                   # Phikon histology features: one row per
                                                      #   spot, index = Visium barcode
                                                      #   ("AAACAACGAATAGTTC-1"), 1024 float
                                                      #   columns.
  1-DLPFC_annotations/
      <section>_truth.txt                             # ground truth, TAB separated, no header:
                                                      #   "<barcode>\t<layer>". Spots without a
                                                      #   layer must be absent (or NA) - they
                                                      #   are dropped on load.

  Sources: counts + images from spatialLIBD / HumanPilot (its 10X folder);
  annotations are HumanPilot's `*_truth.txt`. Section ids are the 12 below.

--------------------------------------------------------------------------------
BACKBONE EMBEDDINGS (one folder per spatial-transcriptomics backbone)
--------------------------------------------------------------------------------
UNIFY never retrains a backbone - it consumes a frozen, pre-computed embedding.
Required data structure:

Data/<BACKBONE>_Results/embeddings/
    <section>_embedding.npy   # float32 [n_spots, d_backbone] - the embedding itself
    <section>_barcodes.npy    # str/object [n_spots] - Visium barcodes in the SAME ROW
                              #   ORDER as the embedding
    <section>_spatial.npy     # float32 [n_spots, 2] - spot coordinates, SAME ROW ORDER.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

# The standard 12 DLPFC sections.
DLPFC_SECTIONS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]


# ============================================================================
# Section loading
# ============================================================================

def load_dlpfc(data_dir, sample_id, backbone, pca_hist,
               hist_filename='phikon_representation.csv', emb_subdir='embeddings'):
    """Load one DLPFC section: backbone embedding, Phikon histology, coords, labels.

    Args:
        data_dir: root ``Data/`` folder.
        sample_id: section id, e.g. '151673'.
        backbone: folder prefix before '_Results' under data_dir
            Embeddings are read from Data/<backbone>_Results/<emb_subdir>/.
        pca_hist: PCA dimensionality applied to the Phikon histology features.
        hist_filename: 'phikon_representation.csv'
        emb_subdir: embedding subfolder.

    Returns:
        B, H, coords, labels, barcodes, label_names
    """
    emb_dir = os.path.join(data_dir, f'{backbone}_Results', emb_subdir)
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(
            f'Backbone embeddings not found: {emb_dir}\n'
            f'Expected {sample_id}_embedding.npy / _barcodes.npy / _spatial.npy there. '
            f'(see the process.py docstring).'
        )

    emb = np.load(os.path.join(emb_dir, f'{sample_id}_embedding.npy'))
    emb_bc = np.load(os.path.join(emb_dir, f'{sample_id}_barcodes.npy'), allow_pickle=True)
    spatial = np.load(os.path.join(emb_dir, f'{sample_id}_spatial.npy'))
    emb_bc = np.asarray(emb_bc).astype(str)

    # Phikon histology: rows indexed by Visium barcode, 1024 feature columns.
    hist_csv = os.path.join(data_dir, '1-DLPFC', sample_id, 'image_representation', hist_filename)
    df = pd.read_csv(hist_csv, index_col=0)
    hist_bc = df.index.values.astype(str)
    hist = df.values.astype(np.float32)

    # Ground-truth layers: "<barcode>\t<layer>" per line, no header.
    truth_path = os.path.join(data_dir, '1-DLPFC_annotations', f'{sample_id}_truth.txt')
    truth = {}
    with open(truth_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                truth[parts[0]] = parts[1]

    # Align the three sources on the barcodes they all share, in sorted order.
    common = sorted(set(emb_bc) & set(hist_bc) & set(truth.keys()))
    assert len(common) > 0, f'No overlapping barcodes for {sample_id}'

    emb_idx = {b: i for i, b in enumerate(emb_bc)}
    hist_idx = {b: i for i, b in enumerate(hist_bc)}
    bc_to_spatial = {b: spatial[i] for i, b in enumerate(emb_bc)}

    B = emb[np.array([emb_idx[b] for b in common])].astype(np.float32)
    H = hist[np.array([hist_idx[b] for b in common])].astype(np.float32)
    coords = np.array([bc_to_spatial[b] for b in common]).astype(np.float32)

    # Z-score the backbone embedding.
    B = ((B - B.mean(axis=0, keepdims=True))/(B.std(axis=0, keepdims=True) + 1e-8))

    # PCA + Z-score the histology (Phikon 1024 -> pca_hist).
    H = PCA(n_components=pca_hist, random_state=42).fit_transform(H)
    H = ((H - H.mean(axis=0, keepdims=True))/(H.std(axis=0, keepdims=True) + 1e-8))
    H = H.astype(np.float32)

    label_names = sorted(set(truth[b] for b in common))
    label_to_int = {name: i for i, name in enumerate(label_names)}
    labels = np.array([label_to_int[truth[b]] for b in common], dtype=np.int64)

    return B, H, coords, labels, np.array(common), label_names


# ============================================================================
# Visium reading
# ============================================================================

TISSUE_POSITION_FILENAMES = (
    'tissue_positions.parquet',
    'tissue_positions.csv',
    'tissue_positions_list.csv',
    'tissue_positions_list.txt',
)

# 10x's documented column order for every one of those files.
TISSUE_POSITION_COLUMNS = [
    'barcode', 'in_tissue', 'array_row', 'array_col',
    'pxl_row_in_fullres', 'pxl_col_in_fullres',
]


def find_tissue_positions(spatial_dir):
    """Return the path of whichever spot-position file this sample ships.

    Raises FileNotFoundError naming every candidate if none is present.
    """
    for name in TISSUE_POSITION_FILENAMES:
        path = os.path.join(spatial_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f'No spot-position file in {spatial_dir}. Expected one of: '
        f'{", ".join(TISSUE_POSITION_FILENAMES)}'
    )


def read_tissue_positions(spatial_dir):
    """Read the spot positions, whichever filename and header style they use.

    Returns: DataFrame indexed by barcode with the columns
             in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    """
    path = find_tissue_positions(spatial_dir)

    if path.endswith('.parquet'):
        pos = pd.read_parquet(path)
        pos.columns = [str(c) for c in pos.columns]
        if 'barcode' in pos.columns:
            pos = pos.set_index('barcode')
    else:
        with open(path) as f:
            first = f.readline().strip().split(',')
        has_header = True
        if len(first) > 1:
            try:
                int(first[1])
                has_header = False
            except ValueError:
                has_header = True

        if has_header:
            pos = pd.read_csv(path, index_col=0)
            pos.columns = [str(c) for c in pos.columns]
            if list(pos.columns) != TISSUE_POSITION_COLUMNS[1:]:
                pos.columns = TISSUE_POSITION_COLUMNS[1:len(pos.columns) + 1]
        else:
            pos = pd.read_csv(path, header=None, index_col=0)
            pos.columns = TISSUE_POSITION_COLUMNS[1:len(pos.columns) + 1]

    pos.index = pos.index.astype(str)
    pos.index.name = 'barcode'
    return pos


def read_visium_section(section_dir, count_file='filtered_feature_bc_matrix.h5', require_positions=True):
    """Read one Visium section: counts + spot positions in ``obsm['spatial']``.

    Tries ``scanpy.read_visium`` first, so a standard Space Ranger folder is read
    by scanpy itself (images and scalefactors included). When scanpy rejects the
    folder, it falls back to ``read_10x_h5`` plus read_tissue_positions, which
    accepts any of the filenames in TISSUE_POSITION_FILENAMES.

    ``obsm['spatial']`` is written as (pxl_col_in_fullres, pxl_row_in_fullres) to
    match scanpy's own convention, which is the way stored in <section>_spatial.npy.

    Returns: AnnData with var_names made unique and, unless the positions are
    absent and not required, obsm['spatial'] populated.
    """
    import scanpy as sc

    try:
        adata = sc.read_visium(path=section_dir, count_file=count_file)
        adata.var_names_make_unique()
        if 'spatial' in adata.obsm:
            return adata
    except (OSError, KeyError, ValueError):
        adata = None

    if adata is None:
        adata = sc.read_10x_h5(os.path.join(section_dir, count_file))
        adata.var_names_make_unique()

    try:
        pos = read_tissue_positions(os.path.join(section_dir, 'spatial'))
    except FileNotFoundError:
        if require_positions:
            raise
        return adata 
    missing = [b for b in adata.obs_names if b not in pos.index]
    if missing:
        raise ValueError(
            f'{len(missing)} barcode(s) from {count_file} are absent from the spot-position '
            f'file in {section_dir} (e.g. {missing[:3]}). The counts and the spatial folder '
            f'must come from the same section.'
        )

    pos = pos.loc[adata.obs_names]
    for col in pos.columns:
        adata.obs[col] = pos[col].values
    adata.obsm['spatial'] = pos[['pxl_col_in_fullres','pxl_row_in_fullres']].to_numpy()
    return adata


def _atomic_write(path, writer):
    """Write via a temp file + os.replace, so a crashed or concurrent run cannot
    leave a half-written graph behind.

    Returns True on success, False when the location is not writable.
    """

    root, ext = os.path.splitext(path)
    tmp = f'{root}.tmp{os.getpid()}{ext}'
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        writer(tmp)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def _normalize_adjacency(A_p):
    """D^{-1/2} (A_p + I) D^{-1/2}; a deterministic function of A_p, so the
    cache stores only A_p and this is recomputed on load."""
    A_hat = A_p + np.eye(A_p.shape[0], dtype=np.float32)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1) + 1e-8))
    return (D_inv_sqrt @ A_hat @ D_inv_sqrt).astype(np.float32)


# ============================================================================
# Co-expression graph
# ============================================================================

def build_coexpr_adjacency(data_dir, sample_id, k, common_barcodes, n_top_genes=3000,
                       n_pcs=50, cache=True, verbose=True):
    """Load the co-expression KNN graph, or compute and cache it, aligned to barcodes.

    Returns:
        A_e [n_common, n_common] array.
    """
    cache_dir = os.path.join(data_dir, '1-DLPFC', sample_id, 'spatial')
    npz_path = os.path.join(cache_dir, f'{sample_id}_A_coexpr_k{k}.npz')
    bc_path = os.path.join(cache_dir, f'{sample_id}_A_coexpr_k{k}_barcodes.npy')

    if cache and os.path.exists(npz_path) and os.path.exists(bc_path):
        import scipy.sparse as sp
        A_dense = sp.load_npz(npz_path).toarray().astype(np.float32)
        src_barcodes = np.load(bc_path, allow_pickle=True).astype(str)
        source = 'cache'
    else:
        A_dense, src_barcodes = _compute_coexpr_graph(
            data_dir, sample_id, k, n_top_genes=n_top_genes, n_pcs=n_pcs)
        source = 'computed'
        if cache:
            import scipy.sparse as sp
            ok = _atomic_write(npz_path, lambda p: sp.save_npz(p, sp.csr_matrix(A_dense)))
            ok = _atomic_write(bc_path, lambda p: np.save(p, src_barcodes)) and ok
            source = 'computed + cached' if ok else 'computed (cache dir not writable)'

    if verbose:
        print(f'    A_e  ({sample_id}, k={k}): {source}')

    # Reindex from the count matrix's barcode order -> common_barcodes order.
    src_idx = {b: i for i, b in enumerate(src_barcodes)}
    missing = [b for b in common_barcodes if b not in src_idx]
    assert not missing, (
        f'{len(missing)} barcode(s) used for training are absent from the co-expression '
        f'graph of {sample_id} (e.g. {missing[:3]}). The counts h5, the embedding and '
        f'the annotations must all come from the same section.'
    )
    indices = np.array([src_idx[b] for b in common_barcodes])
    return A_dense[indices][:, indices]


def _compute_coexpr_graph(data_dir, sample_id, k, n_top_genes=3000, n_pcs=50):
    """Co-expression graph from the counts h5.

    Returns: (A_dense [n, n] float32, barcodes [n] str) in count-matrix order.
    """
    import scanpy as sc
    import scipy.sparse as sp

    # ---get_data_DLPFC ---
    adata = read_visium_section(os.path.join(data_dir, '1-DLPFC', sample_id), require_positions=False)
    ann = pd.read_csv(
        os.path.join(data_dir, '1-DLPFC_annotations', f'{sample_id}_truth.txt'),
        sep='\t', header=None, index_col=0,
    )
    ann.columns = ['region']
    adata.obs['region'] = ann.loc[adata.obs_names, 'region']

    # ---process_data---
    adata = adata[~pd.isnull(adata.obs['region'])]
    try:
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_genes)
    except ImportError as exc:
        raise ImportError(
            "The co-expression graph needs scanpy's seurat_v3 HVG flavour, which "
            "imports skmisc.loess. Install it with:  pip install scikit-misc\n"
            f'(original error: {exc})'
        ) from exc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]

    # ---PCA once on the processed expression---
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    Z_pca = PCA(n_components=n_pcs, random_state=42).fit_transform(X)
    barcodes = np.array(adata.obs_names, dtype=str)

    # ---symmetric binary KNN co-expression graph---
    A_knn = kneighbors_graph(Z_pca, n_neighbors=k, mode='connectivity', include_self=False)
    A_e = A_knn + A_knn.T
    A_e[A_e > 1] = 1.0
    return A_e.toarray().astype(np.float32), barcodes


# ============================================================================
# Spatial graphs
# ============================================================================

def build_spatial_adjacency(coords, k):
    """Build the symmetric KNN spatial adjacency matrix and its normalized form.

    Returns:
        A_p, A_p_norm (D^{-1/2} (A + I) D^{-1/2})
    """
    A_sparse = kneighbors_graph(coords, n_neighbors=k, mode='connectivity', include_self=False)
    A_sparse = A_sparse + A_sparse.T
    A_sparse[A_sparse > 1] = 1

    A_p = np.array(A_sparse.todense(), dtype=np.float32)
    return A_p, _normalize_adjacency(A_p)


def build_multihop_adjacency(A_p):
    """Precompute 1,2,3-hop binary adjacency matrices (with self-loops).

    Returns:
        A_s1, A_s2, A_s3 (binary adjacency for 1/2/3-hop neighborhoods)
    """
    from scipy import sparse

    N = A_p.shape[0]
    A_sp = sparse.csr_matrix(A_p) + sparse.eye(N, dtype=np.float32)
    A_sp = (A_sp > 0).astype(np.float32)
    A_s1 = A_sp
    A_s2 = (A_s1 @ A_s1 > 0).astype(np.float32)
    A_s3 = (A_s2 @ A_s1 > 0).astype(np.float32)
    return A_s1.toarray(), A_s2.toarray(), A_s3.toarray()


# ============================================================================
# Section preparation
# ============================================================================

def prepare_section(data_dir, sample_id, device, backbone, pca_hist,
                    k_neighbors, hist_filename='phikon_representation.csv',
                    emb_subdir='embeddings', cache_coexpr=True, verbose=True):
    """Load a section and build every tensor UNIFY needs, on ``device``.

    The co-expression graph comes from the cache under
    Data/1-DLPFC/<sid>/spatial/ when it is there, and is generated and written
    there when it is not; cache_coexpr=False recomputes it and writes nothing.
    The spatial and multi-hop graphs are always computed.

    Returns a dict with:
        B, H, A_p_norm, A_p, A_e, A_scales,
        labels, coords, barcodes, n_labels, label_names,
        D, D_h, n_spots
    """
    B, H, coords, labels, barcodes, label_names = load_dlpfc(
        data_dir, sample_id, backbone=backbone, pca_hist=pca_hist,
        hist_filename=hist_filename, emb_subdir=emb_subdir,
    )

    A_p_np, A_p_norm_np = build_spatial_adjacency(coords, k=k_neighbors)
    A_s1_np, A_s2_np, A_s3_np = build_multihop_adjacency(A_p_np)
    A_e_np = build_coexpr_adjacency(data_dir, sample_id, k_neighbors, barcodes,
                                cache=cache_coexpr, verbose=verbose)

    def _t(a):
        return torch.tensor(a, dtype=torch.float32).to(device)

    return {
        'B': _t(B),
        'H': _t(H),
        'A_p': _t(A_p_np),
        'A_p_norm': _t(A_p_norm_np),
        'A_e': _t(A_e_np),
        'A_scales': (_t(A_s1_np), _t(A_s2_np), _t(A_s3_np)),
        'labels': labels,
        'coords': coords,
        'barcodes': barcodes,
        'label_names': label_names,
        'n_labels': len(label_names),
        'D': B.shape[1],
        'D_h': H.shape[1],
        'n_spots': B.shape[0],
    }
