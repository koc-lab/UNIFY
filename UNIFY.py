import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Graph Convolutional Layer: H' = sigma(A_p_norm @ H @ W)."""

    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, h, A_p_norm):
        h = A_p_norm @ h
        h = self.linear(h)
        if self.activation:
            h = F.gelu(h)
        return h


class UNIFY(nn.Module):
    """
    Args:
        D: width of the frozen backbone embedding.
        D_h: width of the PCA'd Phikon features.
        n_gcn_layers: GCN layers applied to the suppressed histology.
        D_k: key/query dimension of neighborhood attention.
        D_a: dimension of the inconsistency descriptor.

    Set ``model.bypass_fusion = True`` to return the raw backbone embedding
    (the gate (spatial verification weight)=0 / pure-genomics baseline).
    """

    def __init__(self, D, D_h, n_gcn_layers, D_k, D_a):
        super().__init__()
        self.D = D
        self.D_k = D_k
        self.D_a = D_a
        self.bypass_fusion = False

        # Histology projection: Phikon -> the backbone's native space
        self.W_h = nn.Linear(D_h, D)

        # Multi-scale local representation inconsistency combiner
        self.W_a = nn.Linear(3 * D, D_a)

        self.W_alpha = nn.Linear(D_a, D)

        # GCN stack on the routed morphology features
        self.hist_gcn = nn.ModuleList([
            GCNLayer(D, D, activation=(i < n_gcn_layers - 1))
            for i in range(n_gcn_layers)
        ])
        self.hist_gcn_norm = nn.LayerNorm(D)

        #Cross-modal attention projections
        self.W_Q = nn.Linear(D, D_k, bias=False)
        self.W_K = nn.Linear(D, D_k, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)

        #Final gate MLP over [B || c || a]
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * D + D_a, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.Sigmoid(),
        )

    def forward(self, B, H, A_p_norm, A_scales):
        """
        Args:
            B: [N, D] frozen backbone embedding
            H: [N, D_h] frozen Phikon, PCA'd and z-scored
            A_p_norm: [N, N] normalized spatial adjacency, for the GCN
            A_scales: (A_s1, A_s2, A_s3) binary adjacency matrices with self-loops

        Returns:
            z:       [N, D] the fused embedding
            B:       [N, D] raw genomics
            H_prime: [N, D] GCN-smoothed histology
            g:       [N, D] per-spot per-dimension gate weights
        """
        if self.bypass_fusion:
            dummy_gate = torch.zeros_like(B)  # g = 0 => z = B
            return B, B, B, dummy_gate

        A_s1, A_s2, A_s3 = A_scales

        h_proj = self.W_h(H)

        v_s1 = self._neighborhood_variance(B, A_s1)
        v_s2 = self._neighborhood_variance(B, A_s2)
        v_s3 = self._neighborhood_variance(B, A_s3)
        a = F.gelu(self.W_a(torch.cat([v_s1, v_s2, v_s3], dim=1)))  #Inconsistency descriptor

        alpha = torch.sigmoid(self.W_alpha(a))  # [N, D]
        h_tilde = alpha * h_proj

        H_prime = h_tilde
        h_skip = H_prime
        for gcn_layer in self.hist_gcn:
            H_prime = gcn_layer(H_prime, A_p_norm)
        H_prime = self.hist_gcn_norm(H_prime + h_skip)  # residual + norm

        Q = self.W_Q(B)         # [N, D_k]
        K = self.W_K(H_prime)   # [N, D_k]
        V = self.W_V(H_prime)   # [N, D]
        scores = (Q @ K.t()) / math.sqrt(self.D_k)                             # [N, N]
        mask = (A_s1 > 0) | torch.eye(Q.shape[0], device=Q.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)                                # [N, N]
        c = weights @ V                                                        # [N, D]

        c_norm = F.normalize(c, dim=1)
        neighbor_counts = A_s1.sum(dim=1, keepdim=True).clamp(min=1.0)
        neighbor_c = A_s1 @ c_norm  # [N, D]
        r = (c_norm * neighbor_c).sum(dim=1, keepdim=True) / neighbor_counts   # [N, 1]
        rho = torch.sigmoid(r)  # [N, 1]

        #Gate (Spatial Verification Weight) and Fusion
        gate_raw = self.gate_mlp(torch.cat([B, c, a], dim=1))  # [N, D]
        g = rho * gate_raw 
        z = (1 - g) * B + g * c

        return z, B, H_prime, g

    @staticmethod
    def _neighborhood_variance(B, A_s):
        """Per-dimension variance of B over the neighborhoods defined by A_s."""
        counts = A_s.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (A_s @ B) / counts
        mean_sq = (A_s @ (B ** 2)) / counts
        return (mean_sq - mean ** 2).clamp(min=0.0)


# ============================================================================
# Losses (3-term objective: L_p, L_e, L_reg)
# ============================================================================

def spatial_adjacency_loss(z, A, tau=0.1):
    """Reconstruct a graph (spatial or co-expression) with temperature-scaled cosine.
    """
    z_norm = F.normalize(z, dim=1)
    cosine_sim = z_norm @ z_norm.T
    logits = cosine_sim / tau

    n_pos = A.sum()
    n_neg = A.numel() - n_pos
    pos_weight = (n_neg / (n_pos + 1e-8)).clamp(max=1000.0)

    return F.binary_cross_entropy_with_logits(
        logits, A,
        pos_weight=pos_weight.expand_as(A),
    )


def vicreg_loss(z, lam_var=1.0, lam_cov=0.04):
    """VICReg variance + covariance regularization."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    L_var = torch.clamp(1.0 - std, min=0).mean()

    z_centered = z - z.mean(dim=0)
    n = z.shape[0]
    cov = (z_centered.T @ z_centered) / (n - 1)
    L_cov = cov.pow(2).sum() / z.shape[1] - cov.diagonal().pow(2).sum() / z.shape[1]

    return lam_var * L_var + lam_cov * L_cov


def unify_loss(z, A_p, A_e,
               lambda_p, lambda_e, lambda_reg):
    """Full UNIFY objective.

    Returns: (total_loss, {'p': ..., 'e': ..., 'reg': ...})
    """
    L_p = spatial_adjacency_loss(z, A_p)
    L_e = spatial_adjacency_loss(z, A_e)
    L_reg = vicreg_loss(z)

    total = (lambda_p * L_p
             + lambda_e * L_e
             + lambda_reg * L_reg)
    return total, {'p': L_p, 'e': L_e, 'reg': L_reg}
