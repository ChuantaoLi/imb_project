import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ============================================================
# Parameters and toy geometry from the paper narrative
# ============================================================
k1, k2 = 3, 5
k3 = max(k1, k2)
xi = np.array([0.0, 0.0])

c_all = np.array(
    [
        [0.30, 0.15],
        [-0.25, -0.20],
        [0.10, 0.45],
        [-0.55, 0.35],
        [0.60, -0.40],
        [0.85, 0.20],
        [-0.70, -0.60],
    ]
)

other_all = np.array(
    [
        [-0.10, 0.30],
        [0.40, -0.25],
        [-0.50, -0.15],
        [0.55, 0.40],
        [-0.35, 0.60],
        [0.80, -0.65],
        [-0.85, 0.45],
    ]
)


def sort_by_distance(points):
    dists = np.sqrt(np.sum(points**2, axis=1))
    order = np.argsort(dists)
    return points[order], dists[order]


c_sorted, c_d_sorted = sort_by_distance(c_all)
other_sorted, other_d_sorted = sort_by_distance(other_all)

rk1 = c_d_sorted[k1 - 1]
N_k3_c = c_sorted[:k3]
N_k1_c = c_sorted[:k1]
N_k3_ot = other_sorted[:k3]
T_pts = other_sorted[:k3][other_d_sorted[:k3] <= rk1]

pool = np.vstack([N_k3_c, N_k3_ot])
pool_lbl = np.array(["c"] * len(N_k3_c) + ["o"] * len(N_k3_ot))
pool_d = np.sqrt(np.sum(pool**2, axis=1))
k2_idx = np.argsort(pool_d)[:k2]
N_k2_pts = pool[k2_idx]
N_k2_lbl = pool_lbl[k2_idx]


def contains_point(group, pt):
    return np.any(np.all(np.isclose(group, pt), axis=1))


def make_legend_handles(specs):
    handles = []
    for label, marker, face, edge, size, alpha in specs:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=1.8,
                markersize=size,
                alpha=alpha,
                label=label,
                color=edge,
            )
        )
    return handles


def add_caption(ax, text, color, facecolor):
    ax.text(
        0.5,
        -0.16,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.3,
        color=color,
        bbox=dict(boxstyle="round,pad=0.42", facecolor=facecolor, edgecolor=color, linewidth=1.2),
    )


def draw_geom(
    ax,
    title,
    *,
    show_r_circle=True,
    show_c=True,
    show_other=True,
    highlight_c_k3=False,
    highlight_c_k1=False,
    highlight_conn=False,
    highlight_ot_k3=False,
    highlight_T=False,
    highlight_Fs=False,
    highlight_k2=False,
):
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.92, 0.92)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    if show_r_circle:
        ax.add_patch(
            Circle(
                xi,
                rk1,
                facecolor="#D9ECFA",
                edgecolor="#5B7C99",
                linewidth=1.6,
                linestyle=(0, (4, 3)),
                alpha=0.55,
                zorder=0,
            )
        )
        angle = np.deg2rad(28)
        ax.annotate(
            r"$r_{k_1}(\mathbf{x}_i)$",
            xy=(rk1 * np.cos(angle), rk1 * np.sin(angle)),
            xytext=(0.42, 0.42),
            textcoords="data",
            fontsize=11.5,
            color="#3E5568",
            ha="left",
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#3E5568"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#A9BAC8", alpha=0.95),
        )

    if highlight_Fs and show_r_circle:
        ax.text(
            0.0,
            -0.03,
            r"$F_s(\mathbf{x}_i)=T(\mathbf{x}_i)\cup N_{k_1}^{c}(\mathbf{x}_i)$",
            fontsize=12.0,
            color="#275A84",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#6C99BF", alpha=0.96),
            zorder=9,
        )

    ax.scatter(
        *xi,
        c="#D62728",
        s=260,
        marker="*",
        edgecolors="#7A1111",
        linewidth=1.5,
        zorder=10,
    )

    if show_c:
        for pt in c_all:
            inside_r = np.sqrt(np.sum(pt**2)) <= rk1
            is_k3 = contains_point(N_k3_c, pt)
            is_k1 = contains_point(N_k1_c, pt)
            edge = "white"
            lw = 0.8
            alpha = 1.0 if inside_r else 0.35
            size = 85 if inside_r else 55

            if highlight_c_k3 and is_k3:
                edge, lw, size, alpha = "#F0A202", 2.4, 118, 1.0
            if highlight_c_k1 and is_k1:
                edge, lw, size, alpha = "#16A34A", 2.8, 132, 1.0

            ax.scatter(*pt, c="#1F77B4", s=size, marker="o", edgecolors=edge, linewidth=lw, alpha=alpha, zorder=5)

    if highlight_conn:
        for pt in N_k1_c:
            ax.plot(
                [xi[0], pt[0]],
                [xi[1], pt[1]],
                color="#16A34A",
                linestyle=(0, (4, 3)),
                linewidth=1.6,
                alpha=0.8,
                zorder=2,
            )

    if show_other:
        for pt in other_all:
            inside_r = np.sqrt(np.sum(pt**2)) <= rk1
            is_k3 = contains_point(N_k3_ot, pt)
            is_T = len(T_pts) > 0 and contains_point(T_pts, pt)
            edge = "white"
            lw = 0.8
            alpha = 1.0 if inside_r else 0.38
            size = 82 if inside_r else 52

            if highlight_ot_k3 and is_k3:
                edge, lw, size, alpha = "#F0A202", 2.4, 116, 1.0
            if highlight_T and is_T:
                edge, lw, size, alpha = "#00A6A6", 2.8, 132, 1.0

            ax.scatter(
                *pt,
                c="#C65A46",
                s=size,
                marker="^",
                edgecolors=edge,
                linewidth=lw,
                alpha=alpha,
                zorder=5,
            )

    if highlight_k2:
        for pt, lbl in zip(N_k2_pts, N_k2_lbl):
            marker = "o" if lbl == "c" else "^"
            ax.scatter(*pt, s=280, facecolors="none", edgecolors="#A21CAF", linewidth=2.8, marker=marker, zorder=8)


fig, axes = plt.subplots(3, 2, figsize=(17, 18), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=6 / 72, hspace=0.08, wspace=0.08)

title_pad = r"对每个少数类实例 $\mathbf{x}_i$"

draw_geom(
    axes[0, 0],
    rf"步骤 2.1a：在 $S_c$ 中搜索 $N_{{k_3}}^c(\mathbf{{x}}_i)$" + "\n" + title_pad,
    show_other=False,
    show_r_circle=False,
    highlight_c_k3=True,
)
add_caption(
    axes[0, 0],
    rf"$N_{{k_3}}^c(\mathbf{{x}}_i)$ 表示距 $\mathbf{{x}}_i$ 最近的 {k3} 个同类邻居。" "\n"
    r"它同时为后续构造 $N_{k_1}^{c}(\mathbf{x}_i)$ 与 $N_{k_2}(\mathbf{x}_i)$ 提供候选来源。",
    "#8A5A00",
    "#FFF7DF",
)

draw_geom(
    axes[0, 1],
    rf"步骤 2.1b：保留 $N_{{k_1}}^c(\mathbf{{x}}_i)$ 并记录 $r_{{k_1}}(\mathbf{{x}}_i)$",
    show_other=False,
    highlight_c_k1=True,
    highlight_conn=True,
)
add_caption(
    axes[0, 1],
    rf"$N_{{k_1}}^c(\mathbf{{x}}_i)$ 表示距 $\mathbf{{x}}_i$ 最近的 {k1} 个同类邻居。" "\n"
    rf"$r_{{k_1}}(\mathbf{{x}}_i)$ 是到第 {k1} 个同类邻居的距离，因此此处约为 {rk1:.2f}。",
    "#166534",
    "#EAF7EC",
)

draw_geom(
    axes[1, 0],
    rf"步骤 2.2a：在 $S_{{\tilde c}}$ 中搜索 $N_{{k_3}}^{{\tilde c}}(\mathbf{{x}}_i)$",
    show_c=False,
    highlight_ot_k3=True,
)
add_caption(
    axes[1, 0],
    rf"$N_{{k_3}}^{{\tilde c}}(\mathbf{{x}}_i)$ 表示当前少数类实例 $\mathbf{{x}}_i$" "\n"
    rf"周围最近的 {k3} 个异类邻居。",
    "#8A5A00",
    "#FFF7DF",
)

draw_geom(
    axes[1, 1],
    r"步骤 2.2b：在半径 $r_{k_1}(\mathbf{x}_i)$ 内筛出 $T(\mathbf{x}_i)$",
    show_c=False,
    highlight_T=True,
)
add_caption(
    axes[1, 1],
    r"$T(\mathbf{x}_i)=\{x\in S_{\tilde c}:\Vert x-\mathbf{x}_i\Vert\leq r_{k_1}(\mathbf{x}_i)\}$" "\n"
    rf"也就是保留落在同一半径范围内的异类样本；本例中 $|T(\mathbf{{x}}_i)|={len(T_pts)}$。",
    "#0F766E",
    "#E8FAF8",
)

draw_geom(
    axes[2, 0],
    r"步骤 2.3a：构造第一轮筛选集合 $F_s(\mathbf{x}_i)$",
    highlight_Fs=True,
)
add_caption(
    axes[2, 0],
    r"$F_s(\mathbf{x}_i)=T(\mathbf{x}_i)\cup N_{k_1}^{c}(\mathbf{x}_i)$" "\n"
    r"它表示后续方向评估所使用的第一轮局部候选区域。",
    "#1D4E89",
    "#EAF2FB",
)

draw_geom(
    axes[2, 1],
    r"步骤 2.3b：构造跨类邻域 $N_{k_2}(\mathbf{x}_i)$",
    highlight_k2=True,
)
add_caption(
    axes[2, 1],
    rf"$N_{{k_2}}(\mathbf{{x}}_i)$ 从 $N_{{k_3}}^c(\mathbf{{x}}_i)\cup N_{{k_3}}^{{\tilde c}}(\mathbf{{x}}_i)$ 中" "\n"
    rf"选出整体最近的 {k2} 个点，并作为 NBDOS 的输入邻域。",
    "#7E22CE",
    "#F5ECFC",
)

legend_specs = [
    (r"$\mathbf{x}_i$（当前少数类实例）", "*", "#D62728", "#7A1111", 14, 1.0),
    (r"$x\in S_c$（同类样本）", "o", "#1F77B4", "white", 10, 1.0),
    (r"$x\in S_{\tilde c}$（异类样本）", "^", "#C65A46", "white", 10, 1.0),
    (r"$x\in N_{k_3}^{c}(\mathbf{x}_i)$ 或 $N_{k_3}^{\tilde c}(\mathbf{x}_i)$", "o", "#FFFFFF", "#F0A202", 11, 1.0),
    (r"$x\in N_{k_1}^{c}(\mathbf{x}_i)$", "o", "#FFFFFF", "#16A34A", 11, 1.0),
    (r"$x\in T(\mathbf{x}_i)$", "^", "#FFFFFF", "#00A6A6", 11, 1.0),
    (r"$x\in N_{k_2}(\mathbf{x}_i)$", "o", "#FFFFFF", "#A21CAF", 12, 1.0),
]

fig.legend(
    handles=make_legend_handles(legend_specs),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=3,
    frameon=True,
    fancybox=True,
    framealpha=0.97,
    edgecolor="#D0D7DE",
    fontsize=11,
    handletextpad=0.7,
    columnspacing=1.3,
)

fig.suptitle(
    "SMOM 近邻与候选集构造过程示意图",
    fontsize=18,
    fontweight="bold",
    y=1.02,
)

output_path = r"D:\imb_project\Paper\SMOM_近邻与候选集示意图.png"
ascii_output_path = r"D:\imb_project\Paper\smom_neighbors_candidates_diagram.png"
plt.savefig(ascii_output_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
shutil.copyfile(ascii_output_path, output_path)
print(f"Saved to: {ascii_output_path}")
print(f"Copied to: {output_path}")
