"""Train UNIFY and report ARI / NMI of the fused embedding.


Usage
-----
    python main.py --backbone <NAME> \\
        --pca_hist_dim <int> --k_neighbors <int> --n_gcn_layers <int> \\
        --D_k <int> --D_a <int> \\
        --n_epochs <int> --lr <float> --weight_decay <float> \\
        --lambda_p <float> --lambda_e <float> --lambda_reg <float>

    ... --sections 151673 151674     # a subset instead of all 12
    ... --plot_dir figures/          # 3-panel figure per section
    ... --data_dir /path/to/Data --device 0

Tuned parameters for the GraphSTAR backbone:
    python main.py --backbone GraphSTAR \\
        --pca_hist_dim 256 --k_neighbors 8 --n_gcn_layers 1 \\
        --D_k 32 --D_a 32 \\
        --n_epochs 224 --lr 0.00010718650798690233 \\
        --weight_decay 0.0009850157895612234 \\
        --lambda_p 0.5886331764298722 \\
        --lambda_e 0.9068134323291349 \\
        --lambda_reg 2.6546217615976677

See process.py for the exact `Data/` layout each file must be saved at.
"""

import argparse
import os
import sys
import time
import torch

try:
    from .process import DLPFC_SECTIONS, prepare_section
    from .UNIFY import UNIFY, unify_loss
    from .utils import (evaluate_embedding, get_device, plot_section_clusters, print_table, results_table, seed_everything)
except ImportError: 
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from process import DLPFC_SECTIONS, prepare_section
    from UNIFY import UNIFY, unify_loss
    from utils import (evaluate_embedding, get_device, plot_section_clusters, print_table, results_table, seed_everything)

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')

class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):

    def _get_help_string(self, action):
        if action.required:
            return action.help
        return super()._get_help_string(action)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Train UNIFY and report ARI / NMI of the fused embedding.',
        formatter_class=_HelpFormatter)

    # --- Task ---------------------------------------------------------------
    p.add_argument('--task', default='domain', choices=['domain'],
                   help='evaluation task (domain = spatial domain identification)')

    # --- Data ---------------------------------------------------------------
    p.add_argument('--data_dir', default=DEFAULT_DATA_DIR,
                   help='root Data/ folder (see process.py for the layout)')
    p.add_argument('--backbone', required=True,
                   help='backbone embedding folder prefix: the part of the folder name '
                        'before "_Results" under data_dir, so embeddings are read from '
                        'Data/<backbone>_Results/<emb_subdir>/')
    p.add_argument('--emb_subdir', default='embeddings',
                   help='embedding subfolder')
    p.add_argument('--hist_filename', default='phikon_representation.csv',
                   help='histology CSV under 1-DLPFC/<section>/image_representation/')
    p.add_argument('--sections', nargs='+', default=DLPFC_SECTIONS,
                   help='DLPFC section ids to run')
    p.add_argument('--no_coexpr_cache', action='store_true',
                   help='recompute the co-expression graph every run and write nothing.')

    # --- Device / seed ------------------------------------------------------
    p.add_argument('--device', type=int, default=0, help='CUDA device index (CPU if none)')
    p.add_argument('--seed', type=int, default=0)

    # --- Hyperparameters --------------------------------------------
    p.add_argument('--pca_hist_dim', type=int, required=True,
                   help='PCA dim for the histology features')
    p.add_argument('--k_neighbors', type=int, required=True,
                   help='k for the spatial AND co-expression KNN graphs')
    p.add_argument('--n_gcn_layers', type=int, required=True,
                   help='GCN layers on the suppressed histology')
    p.add_argument('--D_k', type=int, required=True, help='Attention key/query dim')
    p.add_argument('--D_a', type=int, required=True, help='Inconsistency descriptor dim')
    p.add_argument('--n_epochs', type=int, required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--weight_decay', type=float, required=True)
    p.add_argument('--lambda_p', type=float, required=True)
    p.add_argument('--lambda_e', type=float, required=True)
    p.add_argument('--lambda_reg', type=float, required=True)

    # --- Evaluation --------------------------------------------------------
    p.add_argument('--refine_radius', type=int, default=50,
                   help='neighbor count for the majority-vote refinement (0 disables it)')

    # --- Output ------------------------------------------------------------
    p.add_argument('--save_csv', default=None, help='optional path to write the results table')
    p.add_argument('--save_emb_dir', default=None, help='optional folder to save each <section>_unify_embedding.npy')
    p.add_argument('--plot_dir', default=None, help='optional folder for a per-section 3-panel figure')
    p.add_argument('--plot_show', action='store_true', help='display each figure as well as (or instead of) saving it')

    return p.parse_args(argv)


def train(model, data, args):
    """Full-batch training with the 3-loss objective.

    Returns (z, mean_gate).
    """
    seed_everything(args.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=1e-6)

    B, H = data['B'], data['H']
    A_p_norm, A_scales = data['A_p_norm'], data['A_scales']
    A_p, A_e = data['A_p'], data['A_e']

    for epoch in range(args.n_epochs):
        model.train()
        z, B_out, H_prime, g = model(B, H, A_p_norm, A_scales)

        loss, parts = unify_loss(
            z, A_p, A_e,
            lambda_p=args.lambda_p,
            lambda_e=args.lambda_e,
            lambda_reg=args.lambda_reg,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == args.n_epochs - 1:
            print(f'    epoch {epoch + 1:3d}/{args.n_epochs}  loss={loss.item():.4f}  '
                  f'(p={parts["p"].item():.4f} e={parts["e"].item():.4f} '
                  f'reg={parts["reg"].item():.4f})  gate={g.mean().item():.3f}')

    model.eval()
    with torch.no_grad():
        z, _, _, g_final = model(B, H, A_p_norm, A_scales)
    return z, g_final.mean().item()


def run_section(sample_id, args, device):
    """Load, train and evaluate one section. Returns a per-section result dict."""
    t0 = time.time()

    seed_everything(args.seed)
    data = prepare_section(
        args.data_dir, sample_id, device,
        backbone=args.backbone, pca_hist=args.pca_hist_dim,
        k_neighbors=args.k_neighbors, hist_filename=args.hist_filename,
        emb_subdir=args.emb_subdir, cache_coexpr=not args.no_coexpr_cache,
    )
    print(f'  {data["n_spots"]} spots, {data["n_labels"]} clusters | '
          f'{args.backbone}: ({data["n_spots"]}, {data["D"]}), '
          f'Phikon(PCA): ({data["n_spots"]}, {data["D_h"]})')

    seed_everything(args.seed)
    model = UNIFY(
        D=data['D'], D_h=data['D_h'],
        n_gcn_layers=args.n_gcn_layers, D_k=args.D_k, D_a=args.D_a,
    ).to(device)

    z, mean_gate = train(model, data, args)

    metrics = evaluate_embedding(
        z, data['labels'], data['coords'], data['n_labels'],
        refine_radius=args.refine_radius,
    )

    # --- Figure: ground truth | backbone | backbone + UNIFY -----------------
    if args.plot_dir or args.plot_show:
        baseline = evaluate_embedding(
            data['B'], data['labels'], data['coords'], data['n_labels'],
            refine_radius=args.refine_radius,
        )
        save_path = (os.path.join(args.plot_dir, f'{sample_id}_{args.backbone}_unify.png')
                     if args.plot_dir else None)
        plot_section_clusters(
            data['coords'], data['labels'],
            baseline['pred_refined'], metrics['pred_refined'],
            label_names=data['label_names'], section_id=sample_id,
            backbone_name=args.backbone,
            metrics_backbone=baseline, metrics_unify=metrics,
            save_path=save_path, show=args.plot_show,
        )
        print(f'  backbone alone -> mclust ARI={baseline["ari"]:.4f}  '
              f'NMI={baseline["nmi"]:.4f}')
        if save_path:
            print(f'  figure -> {save_path}')

    if args.save_emb_dir:
        import numpy as np
        os.makedirs(args.save_emb_dir, exist_ok=True)
        np.save(os.path.join(args.save_emb_dir, f'{sample_id}_unify_embedding.npy'),
                z.detach().cpu().numpy())
        np.save(os.path.join(args.save_emb_dir, f'{sample_id}_unify_barcodes.npy'),
                data['barcodes'])

    print(f'  {sample_id} -> mclust ARI={metrics["ari"]:.4f}  '
          f'NMI={metrics["nmi"]:.4f}  (unrefined ARI={metrics["ari_raw"]:.4f})  '
          f'gate={mean_gate:.3f}  [{time.time() - t0:.1f}s]\n')

    return {'ARI': metrics['ari'], 'NMI': metrics['nmi'], 'mean_gate': mean_gate}


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isdir(args.data_dir):
        raise SystemExit(
            f'Data dir not found: {args.data_dir}\n'
            f'No data ships with this repository - create it and populate it as '
            f'documented at the top of process.py, then pass --data_dir.')

    device = get_device(args.device)
    print(f'Task: spatial domain identification  |  Backbone: {args.backbone}  |  '
          f'Device: {device}')
    print(f'Arch: {args.pca_hist_dim}-PCA hist, k={args.k_neighbors}, '
          f'{args.n_gcn_layers} GCN, D_k={args.D_k}, D_a={args.D_a}')
    print(f'Train: {args.n_epochs} epochs, lr={args.lr:.6f}, wd={args.weight_decay:.2e}')
    print(f'Loss:  L_p={args.lambda_p:.3f}, L_e={args.lambda_e:.3f}, '
          f'L_reg={args.lambda_reg:.3f}\n')

    results = {}
    for sid in args.sections:
        print('=' * 56)
        print(f'  Section {sid} - training {args.n_epochs} epochs')
        print('=' * 56)
        results[sid] = run_section(sid, args, device)

    table = results_table(results, order=args.sections)
    print_table(table, title=f'UNIFY final-epoch results ({args.backbone})')

    if args.save_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)
        table.to_csv(args.save_csv, index=False)
        print(f'\nSaved results to {args.save_csv}')

    return table


if __name__ == '__main__':
    main()
