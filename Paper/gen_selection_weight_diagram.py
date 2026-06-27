import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ============================================================
# One strict toy example for the generic weight-computation path
# ============================================================
xi = np.array([0.0, 0.0])  # trapped instance

# candidate neighbours in N_{k1}^c(x_i)
xj = np.array([0.22, -0.56])  # the direction currently being evaluated
nk1_other = np.array([[-0.56, 0.10], [0.60, 0.14]])
Nk1 = np.vstack([xj, nk1_other])

# same-class points around x_i
same_extra = np.array(
    [
        [0.08, 0.46],
        [-0.36, 0.24],
        [0.54, -0.08],
        [-0.18, -0.62],
        [0.42, -0.42],
    ]
)

# cross-class points around x_i
other_all = np.array(
    [
        [-0.12, 0.24],
        [0.16, 0.18],
        [-0.28, -0.10],
        [0.26, -0.18],
        [0.36, -0.34],
        [0.48, -0.52],
        [0.06, -0.70],
        [-0.06, -0.48],
        [0.78, 0.18],
        [-0.74, 0.34],
    ]
)

c_all = np.vstack([Nk1, same_extra])
d_xi_xj = np.linalg.norm(xj - xi)
rk1 = np.max(np.sqrt(np.sum((Nk1 - xi) ** 2, axis=1)))

# From previous figure: F_s(x_i) = T(x_i) ∪ N_{k1}^c(x_i)
T_points = other_all[np.sqrt(np.sum((other_all - xi) ** 2, axis=1)) <= rk1 + 1e-9]
Fs_points = np.vstack([Nk1, T_points])

# Step 5.3a: second filtering
Fs_dists = np.sqrt(np.sum((Fs_points - xi) ** 2, axis=1))
Ss_points = Fs_points[Fs_dists <= d_xi_xj + 1e-9]

# Step 5.3b: PN neighborhood with midpoint center and half-distance radius
pn_center = (xi + xj) / 2
pn_radius = d_xi_xj / 2
NPN_points = Ss_points[np.sqrt(np.sum((Ss_points - pn_center) ** 2, axis=1)) <= pn_radius + 1e-9]

# Toy class statistics inside NPN for formula (2)
# class c count
tau_c = 2
# other minority-class count
tau_mi = 1
# majority-class count
tau_ma = 2
E_mi = 1.00
E_ma = 0.50

w1 = 0.5
w2 = 0.3
r1 = 0.4
r2 = 0.2
sw = 1 / np.e + w1 * (r1 * tau_mi / tau_c + r2 * E_mi) + w2 * (r1 * tau_ma / tau_c + r2 * E_ma) + (
    1 - tau_ma / (tau_mi + tau_c)
)


def make_legend_handles(specs):
    handles = []
    for label, marker, face, edge, size, alpha, ls in specs:
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle=ls,
                linewidth=1.8 if ls != "None" else 0.0,
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
        fontsize=11.1,
        color=color,
        bbox=dict(boxstyle="round,pad=0.42", facecolor=facecolor, edgecolor=color, linewidth=1.2),
    )


def draw_background(ax):
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.scatter(other_all[:, 0], other_all[:, 1], c="#D26A54", s=64, marker="^", edgecolors="white", linewidth=0.8, zorder=2)
    ax.scatter(c_all[:, 0], c_all[:, 1], c="#1F77B4", s=76, marker="o", edgecolors="white", linewidth=0.8, zorder=3)
    ax.scatter(*xi, c="#D62728", s=260, marker="*", edgecolors="#7A1111", linewidth=1.5, zorder=10)


def highlight_xj(ax):
    ax.scatter(*xj, c="#1F77B4", s=160, marker="o", edgecolors="#F97316", linewidth=3.0, zorder=9)
    ax.plot([xi[0], xj[0]], [xi[1], xj[1]], color="#F97316", linewidth=1.8, linestyle=(0, (4, 3)), alpha=0.95, zorder=4)
    ax.text(xj[0] + 0.03, xj[1] + 0.04, r"$\mathbf{x}_j$", fontsize=11.5, color="#EA580C", fontweight="bold")


fig, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=6 / 72, hspace=0.08, wspace=0.08)

# ------------------------------------------------------------
# Panel 1: inherit F_s from previous figure
# ------------------------------------------------------------
ax = axes[0, 0]
draw_background(ax)
ax.set_title(r"步骤 5.3 起点：在前图已构造好的 $\mathbf{x}_i.F_s$ 中选择一个候选方向", fontsize=14, fontweight="bold", pad=10)
ax.add_patch(
    Circle(
        xi,
        rk1,
        facecolor="#D9ECFA",
        edgecolor="#7DA5C7",
        linewidth=1.6,
        linestyle=(0, (4, 3)),
        alpha=0.55,
        zorder=0,
    )
)
ax.text(
    -0.10,
    -0.06,
    r"$F_s(\mathbf{x}_i)=T(\mathbf{x}_i)\cup N_{k_1}^{c}(\mathbf{x}_i)$",
    fontsize=11.8,
    color="#275A84",
    bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#6C99BF", alpha=0.96),
)
for pt in nk1_other:
    ax.scatter(*pt, c="#1F77B4", s=120, marker="o", edgecolors="#A21CAF", linewidth=2.2, alpha=0.95, zorder=8)
highlight_xj(ax)
ax.text(
    0.02,
    0.88,
    r"$\mathbf{x}_j\in N_{k_1}^{c}(\mathbf{x}_i)$，不是从 $\mathbf{x}_i.N_{k_2}$ 中选出的",
    transform=ax.transAxes,
    fontsize=11.2,
    color="#7C2D12",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF7EF", edgecolor="#D7A27C", alpha=0.97),
)
add_caption(
    ax,
    r"承接前两张图：$\mathbf{x}_i$ 是被困实例，且 $\mathbf{x}_i.F_s$ 已经由 $T(\mathbf{x}_i)$ 与 $N_{k_1}^{c}(\mathbf{x}_i)$ 构成。"
    "\n"
    r"$\mathbf{x}_i.N_{k_2}$ 只用于 NBDOS 聚类；到了权重计算阶段，论文明确固定 $\mathbf{x}_j\in N_{k_1}^{c}(\mathbf{x}_i)$ 并从 $\mathbf{x}_i.F_s$ 开始筛选。",
    "#1D4E89",
    "#EAF2FB",
)

# ------------------------------------------------------------
# Panel 2: F_s -> S_s(x_j)
# ------------------------------------------------------------
ax = axes[0, 1]
draw_background(ax)
ax.set_title(r"步骤 5.3a：从 $\mathbf{x}_i.F_s$ 中筛出 $\mathbf{x}_i.S_s(\mathbf{x}_j)$", fontsize=14, fontweight="bold", pad=10)
ax.add_patch(
    Circle(
        xi,
        rk1,
        facecolor="#D9ECFA",
        edgecolor="#7DA5C7",
        linewidth=1.6,
        linestyle=(0, (4, 3)),
        alpha=0.55,
        zorder=0,
    )
)
highlight_xj(ax)
for pt in Ss_points:
    is_same = np.any(np.all(np.isclose(c_all, pt), axis=1))
    marker = "o" if is_same else "^"
    ax.scatter(*pt, c="#FFFFFF", s=126 if is_same else 120, marker=marker, edgecolors="#F97316", linewidth=2.1, zorder=8)
ax.text(
    0.06,
    -0.70,
    r"$\mathbf{x}_i.S_s(\mathbf{x}_j)$",
    fontsize=11.6,
    color="#C2410C",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#F5B08A", alpha=0.95),
)
add_caption(
    ax,
    r"按照论文描述，先在 $\mathbf{x}_i.F_s$ 中保留满足 $\Vert x-\mathbf{x}_i\Vert\leq d(\mathbf{x}_i,\mathbf{x}_j)$ 的样本。"
    "\n"
    r"这样得到的集合记为 $\mathbf{x}_i.S_s(\mathbf{x}_j)$，它是后续 PN 判断的候选池，而不是 $k_2$ 近邻集。",
    "#C2410C",
    "#FFF1E8",
)

# ------------------------------------------------------------
# Panel 3: S_s(x_j) -> N_PN
# ------------------------------------------------------------
ax = axes[1, 0]
draw_background(ax)
ax.set_title(r"步骤 5.3b：在 $\mathbf{x}_i.S_s(\mathbf{x}_j)$ 中逐点判断 PN 邻域成员", fontsize=14, fontweight="bold", pad=10)
highlight_xj(ax)
for pt in Ss_points:
    is_same = np.any(np.all(np.isclose(c_all, pt), axis=1))
    marker = "o" if is_same else "^"
    ax.scatter(*pt, c="#FFFFFF", s=116 if is_same else 110, marker=marker, edgecolors="#F5B08A", linewidth=1.5, zorder=6)
ax.add_patch(
    Circle(
        pn_center,
        pn_radius,
        facecolor="#FFE4D8",
        edgecolor="#C2410C",
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        alpha=0.6,
        zorder=1,
    )
)
for pt in NPN_points:
    is_same = np.any(np.all(np.isclose(c_all, pt), axis=1))
    marker = "o" if is_same else "^"
    ax.scatter(*pt, c="#FFFFFF", s=140 if is_same else 132, marker=marker, edgecolors="#C2410C", linewidth=2.3, zorder=9)
ax.text(
    pn_center[0] - 0.17,
    pn_center[1] + 0.16,
    r"$N_{PN}$",
    fontsize=11.8,
    color="#9A3412",
    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#F2B38E", alpha=0.95),
)
ax.text(
    pn_center[0] - 0.24,
    pn_center[1] - 0.30,
    r"$PN(\mathbf{x}_i,\mathbf{x}_j)$",
    fontsize=11.0,
    color="#9A3412",
    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#F2B38E", alpha=0.95),
)
add_caption(
    ax,
    r"然后在 $\mathbf{x}_i.S_s(\mathbf{x}_j)$ 中逐一判断哪些样本真正落在 PN 邻域。"
    "\n"
    r"这里按论文 Fig. 4b 的做法，将 $PN(\mathbf{x}_i,\mathbf{x}_j)$ 画为“以 $\mathbf{x}_i$ 与 $\mathbf{x}_j$ 中点为圆心、$d(\mathbf{x}_i,\mathbf{x}_j)/2$ 为半径”的局部邻域，得到集合 $N_{PN}$。",
    "#9A3412",
    "#FFF1E8",
)

# ------------------------------------------------------------
# Panel 4: compute weight from NPN distribution
# ------------------------------------------------------------
ax = axes[1, 1]
ax.axis("off")
ax.set_title(r"步骤 5.3c：根据 $N_{PN}$ 的类别分布代入公式（2）计算选择权重", fontsize=14, fontweight="bold", pad=10)

ax.text(
    0.06,
    0.86,
    r"$N_{PN}$ 中的类别统计",
    transform=ax.transAxes,
    fontsize=13,
    fontweight="bold",
    color="#1F2937",
)
ax.text(
    0.06,
    0.72,
    r"$\tau_c=2$：PN 邻域中的 class-$c$ 样本数",
    transform=ax.transAxes,
    fontsize=11.8,
    color="#1F77B4",
)
ax.text(
    0.06,
    0.61,
    r"$\tau_{mi}=1$：PN 邻域中的其他少数类样本数",
    transform=ax.transAxes,
    fontsize=11.8,
    color="#C2410C",
)
ax.text(
    0.06,
    0.50,
    r"$\tau_{ma}=2$：PN 邻域中的多数类样本数",
    transform=ax.transAxes,
    fontsize=11.8,
    color="#9A3412",
)
ax.text(
    0.06,
    0.39,
    r"$E_{mi}=1.00,\;E_{ma}=0.50$：对应类别熵",
    transform=ax.transAxes,
    fontsize=11.8,
    color="#7C3AED",
)

formula = (
    r"$\mathbf{x}_i.sw(\mathbf{x}_j)=\frac{1}{e}"
    r"+w_1\!\left(r_1\frac{\tau_{mi}}{\tau_c}+r_2E_{mi}\right)"
    r"+w_2\!\left(r_1\frac{\tau_{ma}}{\tau_c}+r_2E_{ma}\right)"
    r"+\left(1-\frac{\tau_{ma}}{\tau_{mi}+\tau_c}\right)$"
)
ax.text(
    0.06,
    0.18,
    formula,
    transform=ax.transAxes,
    fontsize=11.0,
    color="#5B3A29",
    bbox=dict(boxstyle="round,pad=0.38", facecolor="#FFF7EF", edgecolor="#D7A27C"),
)
ax.text(
    0.63,
    0.54,
    rf"$w_1={w1},\;w_2={w2},\;r_1={r1},\;r_2={r2}$" + "\n" + rf"$\mathbf{{x}}_i.sw(\mathbf{{x}}_j)\approx {sw:.2f}$",
    transform=ax.transAxes,
    ha="center",
    va="center",
    fontsize=12.8,
    color="#166534",
    bbox=dict(boxstyle="round,pad=0.34", facecolor="#EAF7EC", edgecolor="#86C29B"),
)
add_caption(
    ax,
    r"最后，不再看 $\mathbf{x}_i.F_s$ 或 $\mathbf{x}_i.S_s(\mathbf{x}_j)$ 的整体分布，"
    "\n"
    r"而是只使用 $N_{PN}$ 中各类别的数量与熵，严格按论文公式（2）计算方向 $\mathbf{x}_i\!\to\!\mathbf{x}_j$ 的选择权重。",
    "#5B3A29",
    "#FFF7EF",
)

legend_specs = [
    (r"$\mathbf{x}_i$（被困实例）", "*", "#D62728", "#7A1111", 14, 1.0, "None"),
    (r"$x\in S_c$（同类样本）", "o", "#1F77B4", "white", 9, 1.0, "None"),
    (r"$x\in S_{\tilde c}$（异类样本）", "^", "#D26A54", "white", 9, 1.0, "None"),
    (r"$\mathbf{x}_j\in N_{k_1}^{c}(\mathbf{x}_i)$（当前候选方向）", "o", "#1F77B4", "#F97316", 10, 1.0, "None"),
    (r"$x\in \mathbf{x}_i.S_s(\mathbf{x}_j)$ 或 $x\in N_{PN}$", "o", "#FFFFFF", "#C2410C", 0.1, 1.0, "-"),
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
    columnspacing=1.25,
)

fig.suptitle("SMOM 被困实例通用选择权重计算示意图", fontsize=18, fontweight="bold", y=1.02)

output_path = r"D:\imb_project\Paper\SMOM_被困实例通用选择权重示意图.png"
ascii_output_path = r"D:\imb_project\Paper\smom_selection_weight_generic_diagram.png"
plt.savefig(ascii_output_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
shutil.copyfile(ascii_output_path, output_path)
print(f"Saved to: {ascii_output_path}")
print(f"Copied to: {output_path}")
