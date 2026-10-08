"""Shared helpers for UNIFY: seeding, device, clustering, label refinement, metrics.

Nothing here is UNIFY-specific - the same functions score a backbone baseline, so
a fused embedding and a raw backbone embedding always go through the identical
protocol.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(device=0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{device}')
    return torch.device('cpu')


# ============================================================================
# Clustering
# ============================================================================

def cluster_mclust(embedding, num_cluster, modelNames='EEE', random_seed=2020):
    """R mclust (via rpy2) with the 'EEE' model - the field-standard protocol.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    from rpy2.robjects import globalenv
    from rpy2.robjects.vectors import FloatVector, IntVector, StrVector

    robjects.r.library("mclust")
    data_mat = np.array(embedding, dtype=np.float64)

    globalenv['data'] = FloatVector(data_mat.flatten(order='F'))
    globalenv['nr'] = IntVector([data_mat.shape[0]])
    globalenv['nc'] = IntVector([data_mat.shape[1]])
    globalenv['G'] = IntVector([int(num_cluster)])
    globalenv['model_name'] = StrVector([modelNames])

    robjects.r(f'set.seed({random_seed})')
    robjects.r('mat <- matrix(data, nrow=nr, ncol=nc)')
    robjects.r('res <- Mclust(mat, G=G, modelNames=model_name)')
    mclust_res = np.array(robjects.r('res$classification'))
    return mclust_res.astype(int)


# ============================================================================
# Post-processing
# ============================================================================

def refine_labels(labels, coords, radius=50):
    """Reassign each spot to the majority label among its nearest neighbors. It is the standard post-processing across the spatial-transcriptomics literature.
    ``radius`` is a neighbor count, not a distance.
    """
    from scipy.spatial.distance import cdist

    distance = cdist(coords, coords, metric='euclidean')
    labels = np.asarray(labels)
    refined = []
    for i in range(len(labels)):
        neighbors_idx = distance[i].argsort()[1:radius + 1]
        neighbor_labels = labels[neighbors_idx]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        refined.append(values[counts.argmax()])
    return np.array(refined)


# ============================================================================
# Metrics
# ============================================================================

def compute_ari_nmi(true_labels, pred_labels):
    """ARI and NMI of a clustering against the ground truth."""
    return (adjusted_rand_score(true_labels, pred_labels),
            normalized_mutual_info_score(true_labels, pred_labels))


def evaluate_embedding(z, labels, coords, n_labels, refine_radius=50, scale=True):
    """Score an embedding under the spatial-domain-identification protocol.

    z-score -> mclust -> spatial majority-vote refinement -> ARI + NMI. This is
    the protocol every number in the paper uses, so the same call scores a fused
    embedding and a raw backbone embedding.

    Args:
        z: [n_spots, d] embedding.
        labels: [n_spots] ground-truth labels.
        coords: [n_spots, 2] spot coordinates, for the refinement step.
        n_labels: number of clusters to fit (the section's layer count).
        refine_radius: neighbor count for refine_labels; 0 disables refinement.
        scale: z-score the embedding before clustering.

    Returns: dict with ari, nmi, ari_raw, nmi_raw, pred, pred_refined
    """
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    z = np.asarray(z)
    z_in = StandardScaler().fit_transform(z) if scale else z

    pred = cluster_mclust(z_in, n_labels)
    ari_raw, nmi_raw = compute_ari_nmi(labels, pred)

    if refine_radius and refine_radius > 0:
        pred_ref = refine_labels(pred, coords, radius=refine_radius)
    else:
        pred_ref = pred
    ari, nmi = compute_ari_nmi(labels, pred_ref)

    return {
        'ari': ari, 'nmi': nmi,
        'ari_raw': ari_raw, 'nmi_raw': nmi_raw,
        'pred': pred, 'pred_refined': pred_ref,
    }


# ============================================================================
# Reporting
# ============================================================================

def results_table(results, order=None, decimals=4):
    """Build the per-section ARI/NMI table with a MEAN row appended.

    Args:
        results: {section_id: {'ARI': float, 'NMI': float}}
        order: section ids in the order to report (defaults to sorted keys).
    """
    order = list(order) if order is not None else sorted(results)
    table = pd.DataFrame([
        {'Section': sid, 'ARI': results[sid]['ARI'], 'NMI': results[sid]['NMI']}
        for sid in order if sid in results
    ])
    if len(table) == 0:
        return table
    mean_row = pd.DataFrame([{'Section': 'MEAN',
                              'ARI': table['ARI'].mean(),
                              'NMI': table['NMI'].mean()}])
    table = pd.concat([table, mean_row], ignore_index=True)
    table['ARI'] = table['ARI'].round(decimals)
    table['NMI'] = table['NMI'].round(decimals)
    return table


def print_table(table, title=''):
    if title:
        bar = '=' * max(len(title) + 4, 48)
        print(bar)
        print(f'  {title}')
        print(bar)
    try:
        from IPython import get_ipython
        in_notebook = get_ipython() is not None
    except Exception:
        in_notebook = False

    if in_notebook:
        from IPython.display import display
        display(table)
    else:
        print(table.to_string(index=False))


# ============================================================================
# Plotting
# ============================================================================
PAPER_RCPARAMS = {
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.serif': ['CMU Serif', 'Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 17,
    'axes.titlesize': 17,
    'legend.fontsize': 15.5,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'lines.linewidth': 2.6,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 1.0,
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'grid.color': '0.75',
    'axes.axisbelow': True,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
}

PANEL_BACKGROUND = '#EAEAEA'
PANEL_GRID_COLOR = '#969696'
PANEL_SPOT_SIZE = 9.0
PANEL_SPOT_ALPHA = 0.78
PANEL_TITLE_PAD = 8
PANEL_LEGEND_MARKER = 8.0 
PANEL_FIGSIZE = (14.0, 4.8)
PANEL_WSPACE = 0.08
PANEL_MARGINS = dict(left=0.012, right=0.995, top=0.88, bottom=0.04)
PANEL_LEGEND_WIDTH = 0.34
PANEL_FIT_HEIGHT = True


def match_labels(pred, true):
    """Relabel ``pred`` into ``true``'s label space by maximum overlap.

    Cluster ids are arbitrary, so a predicted "cluster 3" has no relation to
    ground-truth layer 3. Solving the assignment problem on the contingency table
    (Hungarian) gives each predicted cluster the ground-truth id it overlaps most,
    which is what lets the three panels share one color scale.

    Returns: int array, same shape as ``pred``.
    """
    from scipy.optimize import linear_sum_assignment

    pred = np.asarray(pred)
    true = np.asarray(true)
    p_vals = np.unique(pred)
    t_vals = np.unique(true)

    overlap = np.zeros((len(p_vals), len(t_vals)), dtype=np.int64)
    for i, p in enumerate(p_vals):
        in_p = pred == p
        for j, t in enumerate(t_vals):
            overlap[i, j] = np.count_nonzero(in_p & (true == t))

    rows, cols = linear_sum_assignment(-overlap)
    mapping = {p_vals[i]: t_vals[j] for i, j in zip(rows, cols)}

    spare = int(t_vals.max()) + 1 if len(t_vals) else 0
    for p in p_vals:
        if p not in mapping:
            mapping[p] = spare
            spare += 1

    return np.array([mapping[p] for p in pred])


def _layer_palette(n_colors):
    import matplotlib.pyplot as plt

    base = list(plt.get_cmap('tab10').colors)
    if n_colors > len(base):
        base = [plt.get_cmap('tab20')(i) for i in range(n_colors)]
    return base[:n_colors]


def _style_panel(ax, coords, codes, palette, title, spot_size):
    ax.set_facecolor(PANEL_BACKGROUND)
    ax.set_axisbelow(True)
    ax.grid(True, color=PANEL_GRID_COLOR, linestyle=':',
            linewidth=0.45, alpha=0.55)
    ax.scatter(coords[:, 0], coords[:, 1],
               c=[palette[int(code)] for code in codes],
               s=spot_size, alpha=PANEL_SPOT_ALPHA, edgecolors='none',
               rasterized=True)
    ax.set_title(title, fontweight='normal', pad=PANEL_TITLE_PAD)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)


def plot_section_clusters(coords, labels_true, pred_backbone, pred_unify,
                          label_names=None, section_id='', backbone_name='backbone',
                          metrics_backbone=None, metrics_unify=None,
                          spot_size=PANEL_SPOT_SIZE, match_colors=True,
                          save_path=None, show=False):

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    coords = np.asarray(coords)
    labels_true = np.asarray(labels_true)
    pred_backbone = np.asarray(pred_backbone)
    pred_unify = np.asarray(pred_unify)

    if match_colors:
        pred_backbone = match_labels(pred_backbone, labels_true)
        pred_unify = match_labels(pred_unify, labels_true)

    names = list(label_names) if label_names is not None else []
    n_codes = int(max(labels_true.max(), pred_backbone.max(), pred_unify.max())) + 1
    palette = _layer_palette(max(n_codes, len(names)))

    def _ari(metrics):
        return f'\n(ARI={metrics["ari"]:.3f})' if metrics else ''

    panels = [
        (labels_true, f'Slice {section_id}\nGround Truth' if section_id
         else 'Ground Truth'),
        (pred_backbone, f'{backbone_name}{_ari(metrics_backbone)}'),
        (pred_unify, f'{backbone_name} + UNIFY{_ari(metrics_unify)}'),
    ]

    with plt.rc_context(PAPER_RCPARAMS):
        fig = plt.figure(figsize=PANEL_FIGSIZE, dpi=180, facecolor='white')
        grid = fig.add_gridspec(
            1, 4,
            width_ratios=[PANEL_LEGEND_WIDTH, 1, 1, 1],
            wspace=PANEL_WSPACE,
            **PANEL_MARGINS,
        )

        legend_ax = fig.add_subplot(grid[0, 0])
        legend_ax.axis('off')

        panel_axes = []
        for column, (codes, title) in enumerate(panels, start=1):
            ax = fig.add_subplot(grid[0, column])
            _style_panel(ax, coords, codes, palette, title, spot_size)
            panel_axes.append(ax)

        handles = [
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=palette[i], markeredgecolor='none',
                   markersize=PANEL_LEGEND_MARKER, label=name)
            for i, name in enumerate(names)
        ]
        if handles:
            legend_ax.legend(handles=handles, loc='center left', frameon=False,
                             handlelength=0.7, handletextpad=0.35,
                             labelspacing=0.35, borderaxespad=0)

        if PANEL_FIT_HEIGHT and panel_axes:
            spread_x = float(np.ptp(coords[:, 0])) or 1.0
            tissue_aspect = float(np.ptp(coords[:, 1])) / spread_x
            for _ in range(6):
                fig.canvas.draw()
                w_in, h_in = fig.get_size_inches()
                slot = panel_axes[0].get_position(original=True)
                slot_w, slot_h = slot.width * w_in, slot.height * h_in
                target_h = slot_w * tissue_aspect
                if slot_h <= 0 or abs(target_h - slot_h) < 0.01:
                    break
                fig.set_size_inches(w_in, h_in * (target_h / slot_h))

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
            root, ext = os.path.splitext(save_path)
            if ext.lower() != '.pdf':
                fig.savefig(root + '.pdf', bbox_inches='tight', pad_inches=0.05)
        if show:
            plt.show()
        elif save_path:
            plt.close(fig) 

    return fig
