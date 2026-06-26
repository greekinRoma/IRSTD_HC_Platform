
import torch
from torch import nn
import math


def spectral_normalize(X, eps=1e-5):
    """
    Normalize X so that ||X @ X^T||_2 is bounded.
    Uses power iteration to estimate largest singular value.
    """
    with torch.no_grad():
        # X: [B, C, N, D] — we work on the last two dims
        # Estimate spectral norm via 1 power iteration
        v = torch.randn(X.shape[0], X.shape[2], 1, device=X.device, dtype=X.dtype)
        v = v / (v.norm(dim=-2, keepdim=True) + eps)
        XT = X.transpose(-1, -2)
        u = X @ XT @ v
        sigma = u.norm(dim=-2, keepdim=True) + eps
    return X / sigma.unsqueeze(-1)


def orthogonality_error(X, eps=1e-6):
    """
    Compute ||X @ X^T - I||_F / N as a measure of non-orthogonality.
    Returns: scalar in [0, inf). 0 = perfectly orthogonal.

    X: [B, C, N, D] — orthogonality measured over last two dims
    """
    N = X.shape[-2]
    X_norm = X / (X.norm(dim=-1, keepdim=True) + eps)
    gram = X_norm @ X_norm.transpose(-1, -2)
    I = torch.eye(N, device=X.device, dtype=X.dtype).unsqueeze(0).unsqueeze(0)
    error = (gram - I).norm(dim=(-1, -2)) / N
    return error  # [B, C]


class OrthLayer(nn.Module):
    """
    Feature orthogonalization layer.

    Args:
        width, height:  spatial dimensions of the feature map
        channels:        number of feature channels
        N:               number of basis vectors
        L:               number of iterations (for iterative methods)
        method:          'newton_schulz' | 'newton_schulz_inv' | 'svd' | 'qr' | 'original'
        spectral_norm:   whether to apply spectral normalization before iteration
        per_channel_weight: use per-channel instead of scalar learnable weight
    """

    def __init__(
        self,
        width,
        height,
        channels,
        N,
        L=8,
        method='original',
        spectral_norm=True,
        per_channel_weight=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.width = width
        self.height = height
        self.num_polar = 1
        self.N = N + self.num_polar
        self.L = L
        self.method = method
        self.use_spectral_norm = spectral_norm
        self.per_channel_weight = per_channel_weight

        # ---- Positional encoding (frozen) ----
        self.constant_polars = nn.Parameter(
            self.get_positional_encoding_half(
                seq_len=channels, num_elements=width * height
            )
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(-1, -1, self.num_polar, -1),
            requires_grad=False,
        )

        # ---- Learnable weights ----
        if per_channel_weight:
            # Per-channel weight: shape [1, channels, 1, 1]
            init_val = math.exp(0.5) * (math.sqrt(2) - 1.0)
            self.weight = nn.Parameter(torch.full([1, channels, 1, 1], init_val))
        else:
            # Scalar weight (original behavior)
            self.weight = nn.Parameter(
                torch.ones([1, 1, 1, 1]) * math.exp(0.5) * (math.sqrt(2) - 1.0)
            )

        # ---- Identity matrix (buffer for device-aware access) ----
        self.register_buffer('eye', torch.eye(self.N).unsqueeze(0).unsqueeze(0))

        # ---- Activation ----
        self.act = nn.GELU()

        # ---- Track orthogonality for monitoring ----
        self.register_buffer('ortho_error_mean', torch.tensor(0.0))       # final output
        self.register_buffer('ortho_error_pre_mean', torch.tensor(0.0))   # before residual
        self.register_buffer('ortho_error_count', torch.tensor(0.0))

    def get_positional_encoding_half(self, seq_len, num_elements):
        """Sinusoidal positional encoding based on spatial position."""
        d_pos = num_elements // 2
        pe = torch.zeros(seq_len, d_pos * 2)
        position = torch.arange(0, d_pos).float().unsqueeze(0)
        position = position // self.width + position % self.width
        div_term = (
            2
            * math.pi
            * torch.exp(
                torch.arange(0, seq_len).float()
                * -(math.log(10000.0) * 2 / seq_len)
            ).unsqueeze(1)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    # ------------------------------------------------------------------
    #  Orthogonalization methods
    # ------------------------------------------------------------------

    def _newton_schulz(self, X):
        """
        Numerically stable Newton-Schulz iteration for (X @ X^T)^{-1/2}.

        Denman-Beavers coupled iteration (Higham, 2008):
          Y_{k+1} = Y_k @ (3I - Z_k @ Y_k) / 2
          Z_{k+1} = (3I - Z_k @ Y_k) @ Z_k / 2
        with Y_0 = A, Z_0 = I  →  Y → A^{1/2}, Z → A^{-1/2}

        Convergence requires eigenvalues of A in (0, 2).
        We guarantee this via:
          1. Spectral normalization:  A = G / λ_max(G)  →  λ(A) ∈ (0, 1]
          2. Ridge regularization:     A += ε·I           →  λ_min > 0
        """
        with torch.no_grad():
            B, C, N, D = X.shape
            I_mat = self.eye.to(X.device)                     # [1, 1, N, N]
            I_exp = I_mat.expand(B, C, N, N)                  # [B, C, N, N]

            # ---- Gram matrix ----
            G = X @ X.transpose(-1, -2)/N                       # [B, C, N, N]

            # ---- Spectral normalisation: estimate λ_max via power iteration ----
            if self.use_spectral_norm:
                with torch.no_grad():
                    v = torch.randn(B, C, N, 1, device=X.device, dtype=X.dtype)
                    for _ in range(5):                        # 5 iterations → ~1e-3 accuracy
                        v = G @ v
                        v = v / (v.norm(dim=-2, keepdim=True) + 1e-12)
                    # Rayleigh quotient: λ_max ≈ (vᵀ G v) / (vᵀ v)
                    lambda_max = (v.transpose(-1, -2) @ G @ v).squeeze(-1).squeeze(-1)
                    lambda_max = lambda_max.clamp(min=1e-6)   # [B, C]
                scale = lambda_max.unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
            else:
                # Fallback: use trace as a rough upper bound (tr(G) = Σ ||x_i||²)
                scale = G.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) / N
                scale = scale.clamp(min=1e-6)

            # ---- Scale + regularise → eigenvalues in (ε, 1] ⊂ (0, 2) ----
            A = G / scale + 1e-5 * I_exp                      # ridge for numerical safety

            # ---- Denman-Beavers iteration ----
            Y = A
            Z = I_exp.clone()

            for _ in range(self.L):
                T_mat = (3.0 * I_mat - Z @ Y) / 2.0
                Y_new = Y @ T_mat
                Z_new = T_mat @ Z

                # Numerical guard: break early if diverging
                if torch.isnan(Y_new).any() or torch.isinf(Y_new).any():
                    break
                Y, Z = Y_new, Z_new

        # Z ≈ A^{-1/2} ≈ ((G + εI)/λ_max)^{-1/2} = λ_max^{1/2} · (G + εI)^{-1/2}
        # Apply to original X; scale factor λ_max^{1/2} is absorbed by later L2-norm.
        result = Z @ X
        return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

    def _newton_schulz_inv(self, X):
        """
        Numerically stable Newton-Schulz for computing (X @ X^T)^{-1}.

        Uses the iteration:  Z_{k+1} = Z_k @ (2I - A @ Z_k)
        with Z_0 = I / λ_max(A).

        Convergence requires eigenvalues of A in (0, 2).
        """
        B, C, N, D = X.shape
        I_mat = self.eye.to(X.device)
        I_exp = I_mat.expand(B, C, N, N)

        G = X @ X.transpose(-1, -2)                       # [B, C, N, N]

        # ---- Spectral normalisation ----
        if self.use_spectral_norm:
            with torch.no_grad():
                v = torch.randn(B, C, N, 1, device=X.device, dtype=X.dtype)
                for _ in range(5):
                    v = G @ v
                    v = v / (v.norm(dim=-2, keepdim=True) + 1e-12)
                lambda_max = (v.transpose(-1, -2) @ G @ v).squeeze(-1).squeeze(-1)
                lambda_max = lambda_max.clamp(min=1e-6)
            scale = lambda_max.unsqueeze(-1).unsqueeze(-1)
        else:
            scale = G.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1) / N
            scale = scale.clamp(min=1e-6)

        A = G / scale                 # ridge regularised

        # ---- Newton iteration for matrix inverse ----
        # Z_0 = I → Z_{k+1} = Z_k (2I - A Z_k)
        Z = I_exp.clone()
        for _ in range(self.L):
            Z_new = Z @ (2.0 * I_mat - A @ Z)
            if torch.isnan(Z_new).any() or torch.isinf(Z_new).any():
                break
            Z = Z_new

        result = Z @ X
        return torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

    def _svd_ortho(self, X):
        """
        Exact orthogonalization via SVD.
        Sets all singular values to 1.
        X_ortho = U @ V^T  (truly orthogonal, ||X_ortho[i]|| = 1)
        """
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        # U: [B, C, N, N], Vh: [B, C, N, D]
        return U @ Vh

    def _qr_ortho(self, X):
        """
        Exact orthogonalization via QR decomposition (economy mode).

        For X: [N, D] with N < D (more columns than rows):
          X^T = Q @ R   where Q: [D, N] has orthonormal columns
          Q^T: [N, D] has orthogonal rows → Q^T @ Q = I_N

        This gives perfectly orthogonal rows in the feature space.
        """
        B, C, N, D = X.shape
        X_flat = X.reshape(B * C, N, D).transpose(-1, -2)  # [B*C, D, N]
        # Economy QR: Q is [B*C, D, N] with orthonormal columns
        Q, R = torch.linalg.qr(X_flat, mode='reduced')
        # Q^T: [B*C, N, D] — rows are orthogonal
        Q = Q.transpose(-1, -2)
        return Q.reshape(B, C, N, D)

    def Ours(self, X):
        """Original ortho implementation (legacy compatibility)."""
        with torch.no_grad():
            I = self.eye.to(X.device)
            conf = math.sqrt(2)
            T = X @ X.transpose(-1, -2)
            N_val = self.N
            X_curr = I - T / N_val
            M = X_curr
            for _ in range(self.L):
                X_curr = X_curr @ X_curr
                M = M + X_curr * conf
                conf = conf * math.sqrt(2)
        return M @ X * torch.nn.functional.leaky_relu(self.weight) 

    def ortho(self, X):
        """Dispatch to the selected orthogonalization method."""
        if self.method == 'newton_schulz':
            X_ortho = self._newton_schulz(X)
        elif self.method == 'newton_schulz_inv':
            X_ortho = self._newton_schulz_inv(X)
        elif self.method == 'svd':
            X_ortho = self._svd_ortho(X)
        elif self.method == 'qr':
            X_ortho = self._qr_ortho(X)
        elif self.method == 'original':
            X_ortho = self.Ours(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # ---- Learnable residual connection (uniform across all methods) ----
        # weight = torch.nn.functional.leaky_relu(self.weight)

        # # ---- Track orthogonality (detached, eval mode only) ----
        # if not self.training:
        #     with torch.no_grad():
        #         err_pre = orthogonality_error(X_ortho).mean()
        #         self.ortho_error_pre_mean = (
        #             0.99 * self.ortho_error_pre_mean + 0.01 * err_pre
        #         )
        #         final = X_ortho * weight + X
        #         err_post = orthogonality_error(final).mean()
        #         self.ortho_error_mean = (
        #             0.99 * self.ortho_error_mean + 0.01 * err_post
        #         )
        #         self.ortho_error_count += 1

        return X_ortho

    def get_orthogonality_metric(self):
        """Return tracked orthogonality errors as (pre_residual, post_residual)."""
        if self.ortho_error_count > 0:
            return {
                'pre_residual': self.ortho_error_pre_mean.item(),
                'post_residual': self.ortho_error_mean.item(),
            }
        return None

    def forward(self, inputs):
        """
        Args:
            inputs: [B, channels, N, width*height]
        Returns:
            outputs: [B, channels, N+num_polar, width*height] (orthogonalized)
        """
        batch_size = inputs.size(0)
        # Concatenate positional encoding
        inputs = torch.cat(
            [inputs, self.constant_polars.expand(batch_size, -1, -1, -1)], dim=2
        )
        # Pre-normalize
        inputs = torch.nn.functional.normalize(inputs, dim=-1, p=2)
        # Orthogonalize
        outputs = self.ortho(inputs)
        # Post-normalize
        outputs = torch.nn.functional.normalize(outputs, dim=-1, p=2)
        # print(outputs@outputs.transpose(-1,-2))
        return outputs


# ------------------------------------------------------------------
#  Utility: compute and log orthogonality across the model
# ------------------------------------------------------------------

def collect_orthogonality_metrics(model, verbose=True):
    """
    Scan a model for all OrthLayer instances and report their
    orthogonality error metrics.

    Returns:
        dict: {layer_name: {'pre_residual': float, 'post_residual': float}}
    """
    metrics = {}
    for name, module in model.named_modules():
        if isinstance(module, OrthLayer):
            err = module.get_orthogonality_metric()
            if err is not None:
                metrics[name] = err
                if verbose:
                    print(
                        f"  [OrthLayer] {name}: "
                        f"pre_residual={err['pre_residual']:.6f}, "
                        f"post_residual={err['post_residual']:.6f}"
                    )
    if verbose and not metrics:
        print("  (no orthogonality metrics recorded yet — run eval mode first)")
    return metrics


def orthogonality_regularization(model, target_error=0.01, method='newton_schulz'):
    """
    Compute an orthogonality regularization loss across all OrthLayer instances.

    Useful as an auxiliary loss during training: loss = task_loss + lambda * ortho_reg

    Args:
        model: the full model
        target_error: soft threshold below which no penalty is applied
        method: only include layers using this method

    Returns:
        torch.Tensor: scalar regularization loss
    """
    total_reg = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for module in model.modules():
        if isinstance(module, OrthLayer) and module.method == method:
            # Compute current orthogonality error
            # This requires having the last output — we use stored metric
            err = module.get_orthogonality_metric()
            if err is not None:
                total_reg = total_reg + max(0.0, err - target_error)
                count += 1
    if count > 0:
        total_reg = total_reg / count
    return total_reg
