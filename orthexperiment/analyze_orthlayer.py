#!/usr/bin/env python3
"""
OrthLayer 权重与特征相关性分析工具
=====================================
分析 OrthLayer 的：
  1. 权重分布 — self.weight 的值和梯度特性
  2. 位置编码 — constant_polars 的频率/相关性结构
  3. 特征正交性 — Gram 矩阵特征值谱、正交化前后对比
  4. 迭代收敛性 — Denman-Beavers 迭代逐次误差
  5. 逐通道分析 — 不同通道的正交化效果差异

用法:
  python orthexperiment/analyze_orthlayer.py \
      --checkpoint checkpoints/best.pth       # 加载训练好的模型（可选）
  python orthexperiment/analyze_orthlayer.py   # 使用随机初始化的 OrthLayer
"""

import os
import sys
import math
import argparse
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式，不弹窗
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.SFBD_Net.OrthogonalizationLayer.OrthLayer import OrthLayer

OUT_DIR = os.path.join(PROJECT_ROOT, "orthexperiment", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def gram_matrix(X):
    """计算 Gram 矩阵 G = X @ X^T, X: [..., N, D]"""
    return X @ X.transpose(-1, -2)


def eigenvalue_spectrum(G):
    """
    计算 Gram 矩阵特征值谱。
    G: [B, C, N, N] 或 [N, N]
    返回: eigenvalues sorted descending, condition_number
    """
    if G.dim() == 4:
        B, C, N, _ = G.shape
        G_flat = G.reshape(-1, N, N)
    else:
        G_flat = G.unsqueeze(0)

    eigvals_list = []
    for i in range(G_flat.shape[0]):
        e = torch.linalg.eigvalsh(G_flat[i])  # symmetric → eigvalsh
        eigvals_list.append(e)

    eigvals = torch.stack(eigvals_list)  # [total, N]
    eigvals_sorted = eigvals.sort(descending=True).values

    cond = eigvals_sorted[:, 0] / (eigvals_sorted[:, -1].clamp(min=1e-12))
    return eigvals_sorted, cond


def orthogonality_error(X):
    """||X_norm @ X_norm^T - I||_F / N   (0 = 完美正交)"""
    N = X.shape[-2]
    Xn = X / (X.norm(dim=-1, keepdim=True) + 1e-8)
    gram = Xn @ Xn.transpose(-1, -2)
    I = torch.eye(N, device=X.device, dtype=X.dtype)
    err = (gram - I).norm(dim=(-1, -2)) / N
    return err


def effective_rank(eigvals, eps=1e-6):
    """
    基于熵的有效秩:  erank = exp(-Σ p_i log p_i),  p_i = λ_i / Σλ_j
    """
    p = eigvals / (eigvals.sum(dim=-1, keepdim=True) + eps)
    p = p.clamp(min=eps)
    entropy = -(p * p.log()).sum(dim=-1)
    return entropy.exp()


# ---------------------------------------------------------------------------
# 1. 权重分析
# ---------------------------------------------------------------------------

def analyze_weights(layer: OrthLayer, tag: str = ""):
    """分析 self.weight 参数。"""
    print(f"\n{'='*60}")
    print(f"1. WEIGHT ANALYSIS {tag}")
    print(f"{'='*60}")

    w = layer.weight.data
    print(f"  weight shape : {tuple(w.shape)}")
    print(f"  weight value : {w.item():.6f}" if w.numel() == 1 else
          f"  weight min/mean/max : {w.min().item():.4f} / {w.mean().item():.4f} / {w.max().item():.4f}")

    # Residual blend ratio
    w_pos = torch.nn.functional.leaky_relu(w)
    blend = w_pos / (1.0 + w_pos) if w.numel() == 1 else w_pos.mean() / (1.0 + w_pos.mean())
    print(f"  leaky_relu(weight) : {w_pos.item() if w.numel()==1 else w_pos.mean().item():.4f}")
    print(f"  residual blend (ortho / total) : {blend.item():.4f}")
    print(f"  → {blend.item()*100:.1f}% 正交化成分, {(1-blend.item())*100:.1f}% 原始特征")
    return {"weight": w, "leaky_relu_weight": w_pos, "blend_ratio": blend}


# ---------------------------------------------------------------------------
# 2. 位置编码分析
# ---------------------------------------------------------------------------

def analyze_positional_encoding(layer: OrthLayer, tag: str = ""):
    """分析 constant_polars 位置编码。"""
    print(f"\n{'='*60}")
    print(f"2. POSITIONAL ENCODING ANALYSIS {tag}")
    print(f"{'='*60}")

    pe = layer.constant_polars.data  # [1, C, num_polar, D]
    C = pe.shape[1]
    D = pe.shape[3]

    print(f"  shape : {tuple(pe.shape)}  (1, channels={C}, num_polar={layer.num_polar}, D={D})")

    # 自相关: pe[0, c, 0, :] 的频谱
    pe_flat = pe[0, :, 0, :]  # [C, D]
    gram_pe = pe_flat @ pe_flat.T  # [C, C] — 通道间相关性
    eigvals_pe, cond_pe = eigenvalue_spectrum(gram_pe)

    print(f"  PE channel Gram condition number : {cond_pe.mean().item():.2f}")
    print(f"  PE effective rank (entropy)      : {effective_rank(eigvals_pe).mean().item():.2f} / {C}")

    # 空间自相关 (选取第一个通道)
    pe_0 = pe[0, 0, 0, :]  # [D]
    # reshape to 2D if possible
    w = layer.width
    h = layer.height
    if w * h == D:
        pe_2d = pe_0.reshape(w, h)
        print(f"  PE spatial shape : {w}×{h}, mean={pe_2d.mean().item():.4f}, std={pe_2d.std().item():.4f}")

    return {
        "pe": pe, "gram_pe": gram_pe, "eigvals_pe": eigvals_pe,
        "cond_pe": cond_pe, "channels": C
    }


# ---------------------------------------------------------------------------
# 3. 特征正交性分析（核心）
# ---------------------------------------------------------------------------

def analyze_feature_orthogonality(layer: OrthLayer, tag: str = ""):
    """
    前向传播 + 分析正交化前后 Gram 矩阵变化。
    使用随机输入模拟真实特征分布。
    """
    print(f"\n{'='*60}")
    print(f"3. FEATURE ORTHOGONALITY ANALYSIS {tag}")
    print(f"{'='*60}")

    B, C = 4, layer.constant_polars.shape[1]
    N_input = layer.N - layer.num_polar  # 不含 polar 的 basis 数量
    D = layer.width * layer.height

    # 生成模拟输入: 有一定相关性的随机特征
    torch.manual_seed(42)
    X_raw = torch.randn(B, C, N_input, D)
    # 注入相关性: 50% 的成分共享
    common = torch.randn(B, C, 1, D) * 0.3
    X_raw = X_raw + common
    X_raw = torch.nn.functional.normalize(X_raw, dim=-1, p=2)
    X_input = X_raw.clone()

    # ---- 正交化前 ----
    X_pre = torch.cat([
        X_raw,
        layer.constant_polars.expand(B, -1, -1, -1)
    ], dim=2)
    X_pre = torch.nn.functional.normalize(X_pre, dim=-1, p=2)

    G_pre = gram_matrix(X_pre)  # [B, C, N, N]
    eigvals_pre, cond_pre = eigenvalue_spectrum(G_pre)
    err_pre = orthogonality_error(X_pre)

    # ---- 正交化后 ----
    layer.eval()
    with torch.no_grad():
        X_post = layer.forward(X_raw)

    G_post = gram_matrix(X_post)
    eigvals_post, cond_post = eigenvalue_spectrum(G_post)
    err_post = orthogonality_error(X_post)

    # ---- 统计 ----
    N_total = layer.N
    print(f"  input  shape : {tuple(X_raw.shape)}")
    print(f"  output shape : {tuple(X_post.shape)}")
    print(f"  {'':>18}  {'PRE-ortho':>16}  {'POST-ortho':>16}")
    print(f"  {'ortho error':>18}  {err_pre.mean().item():>16.8f}  {err_post.mean().item():>16.8f}")
    print(f"  {'condition num':>18}  {cond_pre.mean().item():>16.2f}  {cond_post.mean().item():>16.2f}")
    print(f"  {'λ_max':>18}  {eigvals_pre[:, 0].mean().item():>16.6f}  {eigvals_post[:, 0].mean().item():>16.6f}")
    print(f"  {'λ_min':>18}  {eigvals_pre[:, -1].mean().item():>16.6f}  {eigvals_post[:, -1].mean().item():>16.6f}")
    print(f"  {'effective rank':>18}  {effective_rank(eigvals_pre).mean().item():>16.2f}  {effective_rank(eigvals_post).mean().item():>16.2f}")

    # 相关性改善
    corr_pre = off_diagonal_mean(G_pre)
    corr_post = off_diagonal_mean(G_post)
    print(f"  {'mean |off-diag|':>18}  {corr_pre.mean().item():>16.8f}  {corr_post.mean().item():>16.8f}")

    return {
        "X_input": X_input, "X_pre": X_pre, "X_post": X_post,
        "G_pre": G_pre, "G_post": G_post,
        "eigvals_pre": eigvals_pre, "eigvals_post": eigvals_post,
        "cond_pre": cond_pre, "cond_post": cond_post,
        "err_pre": err_pre, "err_post": err_post,
        "corr_pre": corr_pre, "corr_post": corr_post,
    }


def off_diagonal_mean(G):
    """Gram 矩阵非对角线元素的平均绝对值（衡量特征相关性）"""
    N = G.shape[-1]
    mask = ~torch.eye(N, device=G.device, dtype=torch.bool)
    off = G[..., mask].abs()
    return off.mean(dim=-1)


# ---------------------------------------------------------------------------
# 4. 迭代收敛性分析
# ---------------------------------------------------------------------------

def analyze_iteration_convergence(layer: OrthLayer, tag: str = ""):
    """
    逐迭代分析 Denman-Beavers 收敛过程。
    """
    print(f"\n{'='*60}")
    print(f"4. ITERATION CONVERGENCE ANALYSIS {tag}")
    print(f"{'='*60}")

    B, C = 2, layer.constant_polars.shape[1]
    N_input = layer.N - layer.num_polar
    D = layer.width * layer.height

    torch.manual_seed(123)
    X_raw = torch.randn(B, C, N_input, D)
    X_raw = torch.nn.functional.normalize(X_raw, dim=-1, p=2)

    # 带 polar
    X_full = torch.cat([
        X_raw,
        layer.constant_polars.expand(B, -1, -1, -1)
    ], dim=2)
    X_full = torch.nn.functional.normalize(X_full, dim=-1, p=2)

    N = layer.N
    I_mat = layer.eye.to(X_full.device)
    I_exp = I_mat.expand(B, C, N, N)

    G = X_full @ X_full.transpose(-1, -2)
    A = G / N

    Y = A.clone()
    Z = I_exp.clone()

    errors = []  # ||Z @ A @ Z^T - I|| per iteration
    Y_traces = []
    Z_traces = []

    for step in range(layer.L):
        # 当前正交化误差
        ZAZ = Z @ A @ Z.transpose(-1, -2)
        cur_err = (ZAZ - I_mat).norm(dim=(-1, -2)).mean().item()
        errors.append(cur_err)

        Y_traces.append(Y.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().item())
        Z_traces.append(Z.diagonal(dim1=-2, dim2=-1).sum(dim=-1).mean().item())

        T_mat = (3.0 * I_mat - Z @ Y) / 2.0
        Y_new = Y @ T_mat
        Z_new = T_mat @ Z

        if torch.isnan(Y_new).any() or torch.isinf(Y_new).any():
            print(f"  → diverged at step {step}")
            break
        Y, Z = Y_new, Z_new

    print(f"  L = {layer.L}")
    print(f"  Step  |  ||ZAZ^T - I||_F")
    for i, e in enumerate(errors):
        marker = " ← converged" if e < 1e-6 else ""
        print(f"  {i:4d}  |  {e:.8e}{marker}")

    # 收敛速度估计
    if len(errors) >= 3 and errors[-1] > 1e-12:
        rates = []
        for i in range(2, len(errors)):
            if errors[i - 1] > 1e-12 and errors[i - 2] > 1e-12:
                r = math.log(errors[i] / errors[i - 1]) / math.log(errors[i - 1] / errors[i - 2])
                rates.append(r)
        if rates:
            print(f"  estimated convergence order : {np.mean(rates):.2f}  (quadratic ≈ 2.0)")

    return {"errors": errors, "Y_traces": Y_traces, "Z_traces": Z_traces, "L": layer.L}


# ---------------------------------------------------------------------------
# 5. 逐通道分析
# ---------------------------------------------------------------------------

def analyze_per_channel(layer: OrthLayer, data: dict, tag: str = ""):
    """逐通道正交化效果差异。"""
    print(f"\n{'='*60}")
    print(f"5. PER-CHANNEL ANALYSIS {tag}")
    print(f"{'='*60}")

    err_pre = data["err_pre"]   # [B, C]
    err_post = data["err_post"]  # [B, C]
    cond_pre = data["cond_pre"]  # [B*C]
    cond_post = data["cond_post"]

    C = err_pre.shape[1]
    err_pre_c = err_pre.mean(dim=0)   # 平均到通道
    err_post_c = err_post.mean(dim=0)
    improvement = err_pre_c - err_post_c

    print(f"  Channel | err_pre    | err_post   | improvement")
    for c in range(min(C, 16)):  # 最多显示 16 个通道
        print(f"  {c:7d} | {err_pre_c[c].item():.6e} | {err_post_c[c].item():.6e} | {improvement[c].item():+.6e}")

    print(f"  ---")
    print(f"  mean improvement across channels : {improvement.mean().item():+.6e}")
    print(f"  std  improvement across channels : {improvement.std().item():.6e}")

    return {"err_pre_c": err_pre_c, "err_post_c": err_post_c, "improvement_c": improvement}


# ---------------------------------------------------------------------------
# 6. 可视化
# ---------------------------------------------------------------------------

def plot_results(all_data: dict, tag: str = ""):
    """生成综合分析图。"""
    tag_suffix = f"_{tag}" if tag else ""
    print(f"\n{'='*60}")
    print(f"6. GENERATING PLOTS ...")
    print(f"{'='*60}")

    # ---------- Fig 1: Gram 矩阵对比 ----------
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"OrthLayer Feature Orthogonality Analysis{tag_suffix.replace('_',' ')}",
                 fontsize=14, fontweight="bold")

    for idx, (key, d) in enumerate(all_data.items()):
        G_pre = d["G_pre"][0, 0].cpu().numpy()   # 第一个 batch, 第一个通道
        G_post = d["G_post"][0, 0].cpu().numpy()

        ax1 = axes[0, idx] if len(all_data) > 1 else axes[0, 0]
        ax2 = axes[1, idx] if len(all_data) > 1 else axes[1, 0]

        if len(all_data) == 1:
            ax1, ax2 = axes[0, 0], axes[1, 0]
        elif len(all_data) <= 3:
            ax1 = axes[0, idx]
            ax2 = axes[1, idx]
        else:
            row, col = divmod(idx, 3)
            ax1 = axes[row * 2, col]
            ax2 = axes[row * 2 + 1, col]

        im1 = ax1.imshow(G_pre, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax1.set_title(f"{key} — Gram PRE-ortho\nmean|off|={d['corr_pre'].mean().item():.4f}")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        im2 = ax2.imshow(G_post, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax2.set_title(f"{key} — Gram POST-ortho\nmean|off|={d['corr_post'].mean().item():.4f}")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

    # 隐藏多余的子图
    for idx in range(len(all_data), 3):
        axes[0, idx].set_visible(False)
        axes[1, idx].set_visible(False)

    plt.tight_layout()
    path1 = os.path.join(OUT_DIR, f"gram_matrix{tag_suffix}.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # ---------- Fig 2: 特征值谱 ----------
    fig, axes = plt.subplots(1, min(len(all_data), 3), figsize=(5 * min(len(all_data), 3) + 2, 5))
    if len(all_data) == 1:
        axes = [axes]
    fig.suptitle("Eigenvalue Spectrum (log scale)", fontsize=13, fontweight="bold")

    for idx, (key, d) in enumerate(all_data.items()):
        if idx >= 3:
            break
        ax = axes[idx]
        ev_pre = d["eigvals_pre"].mean(dim=0).cpu().numpy()
        ev_post = d["eigvals_post"].mean(dim=0).cpu().numpy()
        N = len(ev_pre)

        ax.semilogy(range(N), ev_pre, "o-", ms=4, label="pre-ortho", alpha=0.7)
        ax.semilogy(range(N), ev_post, "s-", ms=4, label="post-ortho", alpha=0.7)
        ax.axhline(y=1.0, color="gray", ls="--", alpha=0.5, label="ideal (all=1)")
        ax.set_xlabel("Eigenvalue index")
        ax.set_ylabel("Eigenvalue (log)")
        ax.set_title(key)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(OUT_DIR, f"eigenvalue_spectrum{tag_suffix}.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path2}")

    # ---------- Fig 3: 正交性误差对比 ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = list(all_data.keys())
    x = np.arange(len(methods))
    width = 0.35

    err_pre_means = [all_data[m]["err_pre"].mean().item() for m in methods]
    err_post_means = [all_data[m]["err_post"].mean().item() for m in methods]

    bars1 = ax.bar(x - width / 2, err_pre_means, width, label="pre-ortho", color="coral", alpha=0.8)
    bars2 = ax.bar(x + width / 2, err_post_means, width, label="post-ortho", color="steelblue", alpha=0.8)

    for bar, val in zip(bars1, err_pre_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(err_pre_means) * 0.02,
                f"{val:.4f}", ha="center", fontsize=8)
    for bar, val in zip(bars2, err_post_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(err_post_means) * 0.02,
                f"{val:.4f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Orthogonality Error  ||X X^T - I|| / N")
    ax.set_title("Orthogonality Error: Pre vs Post", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path3 = os.path.join(OUT_DIR, f"ortho_error_comparison{tag_suffix}.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path3}")

    # ---------- Fig 4: 收敛曲线（仅第一个方法） ----------
    if all_data:
        first_key = list(all_data.keys())[0]
        if "convergence" in all_data[first_key]:
            conv = all_data[first_key]["convergence"]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.semilogy(conv["errors"], "o-", ms=6, color="darkgreen", label=r"$||Z A Z^T - I||_F$")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Error (log scale)")
            ax.set_title(f"Denman-Beavers Convergence ({first_key})", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plt.tight_layout()
            path4 = os.path.join(OUT_DIR, f"convergence{tag_suffix}.png")
            fig.savefig(path4, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path4}")

    # ---------- Fig 5: 逐通道改善 ----------
    if all_data:
        fig, axes = plt.subplots(1, min(len(all_data), 3),
                                 figsize=(5 * min(len(all_data), 3) + 2, 4))
        if len(all_data) == 1:
            axes = [axes]
        for idx, (key, d) in enumerate(all_data.items()):
            if idx >= 3:
                break
            ax = axes[idx]
            imp = d.get("improvement_c", None)
            if imp is not None:
                C = len(imp)
                colors = ["green" if v > 0 else "red" for v in imp]
                ax.bar(range(C), imp.cpu().numpy(), color=colors, alpha=0.8)
                ax.axhline(y=0, color="black", lw=0.5)
                ax.set_xlabel("Channel")
                ax.set_ylabel("Improvement (err_pre − err_post)")
                ax.set_title(f"{key}")
                ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path5 = os.path.join(OUT_DIR, f"per_channel{tag_suffix}.png")
        fig.savefig(path5, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path5}")

    return [path1, path2, path3]


# ---------------------------------------------------------------------------
# 7. 加载 checkpoint（可选）
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: str, device="cpu"):
    """尝试从 checkpoint 加载训练好的模型，提取 OrthLayer。"""
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 尝试多种 checkpoint 格式
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"] if isinstance(ckpt["model"], dict) else ckpt["model"].state_dict()
        else:
            state = ckpt  # 可能直接是 state_dict
    else:
        state = ckpt

    # 筛选 OrthLayer 相关的 key
    orth_keys = {k: v for k, v in state.items() if "orth_layer" in k}
    print(f"  Found {len(orth_keys)} orth_layer-related keys")

    if not orth_keys:
        # 可能 key 名称不同，尝试模糊匹配
        orth_keys = {k: v for k, v in state.items()
                     if any(p in k.lower() for p in ["orth", "ortho", "orthogonal"])}
        print(f"  Fuzzy match: {len(orth_keys)} keys")

    if orth_keys:
        print("  Keys:")
        for k in list(orth_keys.keys())[:20]:
            print(f"    {k}: {tuple(orth_keys[k].shape)}")
    else:
        print("  WARNING: No orth_layer keys found. Will analyze random init.")

    return orth_keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OrthLayer Weight & Feature Correlation Analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--width", type=int, default=32, help="Feature map width")
    parser.add_argument("--height", type=int, default=32, help="Feature map height")
    parser.add_argument("--channels", type=int, default=8, help="Hidden channels")
    parser.add_argument("--N", type=int, default=16, help="Number of basis vectors")
    parser.add_argument("--L", type=int, default=10, help="Iteration count (L)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- 创建 OrthLayer ----
    layer = OrthLayer(
        width=args.width,
        height=args.height,
        channels=args.channels,
        N=args.N,
        L=args.L,
    ).to(device)

    # ---- 如果有 checkpoint，加载权重 ----
    orth_state = {}
    if args.checkpoint:
        orth_state = load_model_from_checkpoint(args.checkpoint, device)
        if orth_state:
            # 尝试加载（key 名称可能需要映射）
            try:
                layer.load_state_dict(orth_state, strict=False)
                print("  Loaded weights into OrthLayer (strict=False)")
            except Exception as e:
                print(f"  Could not load weights: {e}")

    # ---- 执行分析 ----
    all_data = {}

    # 对每种配置进行分析
    configs = [
        ("random_init", layer),
    ]

    # 也可用不同 L 做对比
    if args.L >= 8:
        layer_L4 = OrthLayer(
            width=args.width, height=args.height,
            channels=args.channels, N=args.N, L=4,
        ).to(device)
        configs.append(("L=4", layer_L4))

    for tag, lay in configs:
        print(f"\n{'#'*60}")
        print(f"# Analyzing: {tag}")
        print(f"{'#'*60}")

        analyze_weights(lay, tag)
        pe_data = analyze_positional_encoding(lay, tag)
        ortho_data = analyze_feature_orthogonality(lay, tag)
        conv_data = analyze_iteration_convergence(lay, tag)
        ch_data = analyze_per_channel(lay, ortho_data, tag)

        all_data[tag] = {
            **pe_data, **ortho_data, **ch_data,
            "convergence": conv_data,
        }

    # ---- 生成图表 ----
    plot_results(all_data)

    # ---- 保存数值结果 ----
    summary = {}
    for tag, d in all_data.items():
        summary[tag] = {
            "err_pre_mean": d["err_pre"].mean().item(),
            "err_post_mean": d["err_post"].mean().item(),
            "err_reduction": (d["err_pre"].mean() - d["err_post"].mean()).item(),
            "cond_pre_mean": d["cond_pre"].mean().item(),
            "cond_post_mean": d["cond_post"].mean().item(),
            "effective_rank_pre": effective_rank(d["eigvals_pre"]).mean().item(),
            "effective_rank_post": effective_rank(d["eigvals_post"]).mean().item(),
            "corr_pre_mean": d["corr_pre"].mean().item(),
            "corr_post_mean": d["corr_post"].mean().item(),
        }
        if d.get("convergence"):
            summary[tag]["final_conv_error"] = d["convergence"]["errors"][-1]

    import json
    summary_path = os.path.join(OUT_DIR, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Results directory: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
