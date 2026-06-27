import shutil
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ============================================================
# Parameters
# ============================================================
k2 = 5
rTh = 0.6
nTh = 3

c_all = np.array(
    [
        [-0.65, 0.30],
        [-0.55, 0.42],
        [-0.58, 0.18],
        [-0.42, 0.32],
        [-0.48, 0.22],
        [-0.35, 0.38],
        [-0.52, 0.52],
        [0.42, -0.32],
        [0.58, -0.22],
        [0.52, -0.10],
        [0.62, -0.28],
        [0.48, -0.38],
        [0.78, 0.58],
        [0.85, 0.68],
        [-0.08, -0.28],
        [0.08, -0.42],
        [0.18, -0.22],
        [-0.22, -0.48],
        [-0.78, 0.08],
        [0.72, -0.58],
    ]
)

other_all = np.array(
    [
        [-0.15, -0.20],
        [0.02, -0.35],
        [0.12, -0.15],
        [-0.28, -0.35],
        [0.25, -0.30],
        [-0.05, -0.50],
        [0.30, -0.10],
        [0.68, 0.52],
        [0.90, 0.60],
        [0.72, 0.72],
        [0.70, -0.15],
        [0.35, -0.50],
    ]
)

n_c = len(c_all)
all_pts = np.vstack([c_all, other_all])
all_labels = np.array(["c"] * len(c_all) + ["o"] * len(other_all))

# ============================================================
# Compute cross-class k2-NN used by NBDOS
# ============================================================
k2_nn = []
k2_nn_c_counts = []
for i in range(n_c):
    dists = np.sqrt(np.sum((all_pts - c_all[i]) ** 2, axis=1))
    dists[i] = np.inf
    nn_idx = np.argsort(dists)[:k2]
    k2_nn.append(nn_idx)
    k2_nn_c_counts.append(int(np.sum(all_labels[nn_idx] == "c")))

soft_core = np.array([round(cnt / k2) >= rTh for cnt in k2_nn_c_counts])
sf_idx = np.where(soft_core)[0]

rev_knn = [set() for _ in range(n_c)]
for i in range(n_c):
    for j in k2_nn[i]:
        if j < n_c and j != i:
            rev_knn[j].add(i)

Hkc = [set() for _ in range(n_c)]
for i in range(n_c):
    if not soft_core[i]:
        continue
    Hkc[i].add(i)
    for j in k2_nn[i]:
        if j < n_c:
            Hkc[i].add(j)
    for j in rev_knn[i]:
        if soft_core[j]:
            Hkc[i].add(j)

cluster_id = np.zeros(n_c, dtype=int)
cur_id = 0
for seed in range(n_c):
    if not soft_core[seed] or cluster_id[seed] != 0:
        continue
    cur_id += 1
    queue = deque([seed])
    cluster_id[seed] = cur_id
    while queue:
        xi_idx = queue.popleft()
        for xl in Hkc[xi_idx]:
            if cluster_id[xl] == 0:
                cluster_id[xl] = cur_id
                if soft_core[xl]:
                    queue.append(xl)

cluster_sizes = {cid: int(np.sum(cluster_id == cid)) for cid in range(1, cur_id + 1)}
valid_clusters = {cid for cid, size in cluster_sizes.items() if size >= nTh}
is_Oi = np.array([cluster_id[i] in valid_clusters for i in range(n_c)])
is_Ti = ~is_Oi


def add_caption(ax, text, color, facecolor):
    ax.text(
        0.5,
        -0.17,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.1,
        color=color,
        bbox=dict(boxstyle="round,pad=0.42", facecolor=facecolor, edgecolor=color, linewidth=1.2),
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
                linewidth=1.7 if ls != "None" else 0.0,
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


def draw_base(ax, title):
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-0.92, 0.92)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)


def draw_points(ax, cluster_colors=None, show_other=True, show_trapped_gray=False):
    cluster_colors = cluster_colors or {}

    for i, pt in enumerate(c_all):
        face = "#1F77B4"
        edge = "white"
        size = 85
        alpha = 1.0
        lw = 0.9

        if cluster_id[i] > 0 and cluster_id[i] in valid_clusters:
            face = cluster_colors.get(cluster_id[i], "#1F77B4")
            edge = "#243447"
            lw = 1.5
            size = 96
        elif show_trapped_gray and is_Ti[i]:
            face = "#B9BEC5"
            edge = "#6B7280"
            size = 86
            lw = 1.4
            alpha = 0.95

        ax.scatter(*pt, c=face, s=size, marker="o", edgecolors=edge, linewidth=lw, alpha=alpha, zorder=5)

    if show_other:
        ax.scatter(
            other_all[:, 0],
            other_all[:, 1],
            c="#C65A46",
            s=72,
            marker="^",
            edgecolors="white",
            linewidth=0.8,
            zorder=3,
        )


def draw_knn_edges(ax, *, color_same="#B6BDC7", color_cross="#D5D9DE", alpha_same=0.55, alpha_cross=0.35):
    for i in range(n_c):
        for j in k2_nn[i]:
            pt_i = c_all[i]
            pt_j = all_pts[j]
            if j < n_c:
                ax.plot(
                    [pt_i[0], pt_j[0]],
                    [pt_i[1], pt_j[1]],
                    color=color_same,
                    linewidth=0.8,
                    alpha=alpha_same,
                    zorder=1,
                )
            else:
                ax.plot(
                    [pt_i[0], pt_j[0]],
                    [pt_i[1], pt_j[1]],
                    color=color_cross,
                    linewidth=0.7,
                    linestyle=(0, (1, 2)),
                    alpha=alpha_cross,
                    zorder=1,
                )


cluster_palette = {1: "#3366CC", 2: "#DC3912", 3: "#FF9900"}

fig, axes = plt.subplots(3, 2, figsize=(17, 18), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=6 / 72, hspace=0.08, wspace=0.08)

draw_base(axes[0, 0], r"步骤 3.1：基于 $N_{k_2}(\mathbf{x}_i)$ 判定软核实例")
draw_knn_edges(axes[0, 0])
draw_points(axes[0, 0])
for idx in sf_idx:
    axes[0, 0].scatter(*c_all[idx], s=150, facecolors="#1F77B4", edgecolors="#16A34A", linewidth=2.8, zorder=8)
add_caption(
    axes[0, 0],
    r"当 $\mathrm{round}(|N_{k_2}(\mathbf{x}_i)\cap S_c|/k_2)\geq r_{Th}$ 时，$\mathbf{x}_i$ 被判为软核实例。" "\n"
    rf"本例中 $k_2={k2}$、$r_{{Th}}={rTh}$，因此共有 {len(sf_idx)} 个软核实例。",
    "#166534",
    "#EAF7EC",
)

target = int(sf_idx[0])
gkc_knn_c = {j for j in k2_nn[target] if j < n_c and j != target}
gkc_rev_sf = {j for j in rev_knn[target] if soft_core[j] and j not in gkc_knn_c}

draw_base(axes[0, 1], r"步骤 3.2a：为单个软核实例构造 $G_k^c(\mathbf{x}_i)$")
draw_points(axes[0, 1])
tp = c_all[target]
if len(Hkc[target]) > 1:
    radius = max(np.linalg.norm(c_all[j] - tp) for j in Hkc[target]) + 0.09
    axes[0, 1].add_patch(
        Circle(tp, radius, facecolor="#D7E9F8", edgecolor="#4F81BD", linewidth=1.8, linestyle=(0, (5, 3)), alpha=0.35, zorder=0)
    )
for j in gkc_knn_c:
    axes[0, 1].plot([tp[0], c_all[j][0]], [tp[1], c_all[j][1]], color="#0EA5E9", linewidth=1.6, alpha=0.85, zorder=4)
    axes[0, 1].scatter(*c_all[j], s=138, facecolors="#1F77B4", edgecolors="#0EA5E9", linewidth=2.4, zorder=7)
for j in gkc_rev_sf:
    axes[0, 1].plot([tp[0], c_all[j][0]], [tp[1], c_all[j][1]], color="#A21CAF", linewidth=1.5, linestyle=(0, (4, 2)), alpha=0.9, zorder=4)
    axes[0, 1].scatter(*c_all[j], s=138, facecolors="#1F77B4", edgecolors="#A21CAF", linewidth=2.4, zorder=7)
axes[0, 1].scatter(*tp, s=190, facecolors="#1F77B4", edgecolors="#F97316", linewidth=3.2, zorder=9)
axes[0, 1].text(
    tp[0] + 0.03,
    tp[1] + 0.08,
    r"$\mathbf{x}_i$",
    fontsize=12,
    color="#9A3412",
    weight="bold",
    zorder=10,
)
add_caption(
    axes[0, 1],
    r"$G_k^c(\mathbf{x}_i)=\{\mathbf{x}_i\}\cup\{N_{k_2}(\mathbf{x}_i)\cap S_c\}\cup\{RN_{k_2}(\mathbf{x}_i)\cap \mathrm{sf}\}$" "\n"
    r"蓝色连线表示同类 $k_2$ 近邻，紫色连线表示属于软核实例的逆近邻。",
    "#1D4E89",
    "#EAF2FB",
)

pair_found = None
for i in range(n_c):
    if not soft_core[i]:
        continue
    for j in Hkc[i]:
        if j > i and soft_core[j] and i in Hkc[j]:
            pair_found = (i, j)
            break
    if pair_found is not None:
        break

a, b = pair_found
draw_base(axes[1, 0], r"步骤 3.2b：$G_k^c$ 内部的对称连通关系")
draw_points(axes[1, 0])
axes[1, 0].scatter(*c_all[a], s=190, facecolors="#1F77B4", edgecolors="#F97316", linewidth=3.2, zorder=9)
axes[1, 0].scatter(*c_all[b], s=190, facecolors="#1F77B4", edgecolors="#A21CAF", linewidth=3.2, zorder=9)
axes[1, 0].plot(
    [c_all[a][0], c_all[b][0]],
    [c_all[a][1], c_all[b][1]],
    color="#15803D",
    linewidth=2.0,
    alpha=0.9,
    zorder=6,
)
for j in (Hkc[a] | Hkc[b]) - {a, b}:
    axes[1, 0].scatter(*c_all[j], s=108, facecolors="#1F77B4", edgecolors="#EAB308", linewidth=1.8, zorder=7)
mid = (c_all[a] + c_all[b]) / 2
axes[1, 0].text(
    mid[0],
    mid[1] + 0.08,
    r"$\mathbf{x}_b\in G_k^c(\mathbf{x}_a)$" + "\n" + r"$\mathbf{x}_a\in G_k^c(\mathbf{x}_b)$",
    fontsize=11.2,
    ha="center",
    va="bottom",
    color="#166534",
    bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#86C29B", alpha=0.96),
    zorder=10,
)
add_caption(
    axes[1, 0],
    r"对于软核实例，Lemma 1 保证如下对称关系：" "\n"
    r"$\mathbf{x}_b\in G_k^c(\mathbf{x}_a)\Longleftrightarrow \mathbf{x}_a\in G_k^c(\mathbf{x}_b)$。",
    "#166534",
    "#EAF7EC",
)

seed1 = next(i for i in range(n_c) if cluster_id[i] == 1 and soft_core[i])
layers = {seed1: 0}
queue = deque([seed1])
parents = {}
while queue:
    xi_idx = queue.popleft()
    for xl in Hkc[xi_idx]:
        if xl not in layers and soft_core[xl]:
            layers[xl] = layers[xi_idx] + 1
            parents[xl] = xi_idx
            queue.append(xl)

layer_edge = {0: "#DC2626", 1: "#F97316", 2: "#EAB308", 3: "#FACC15"}
draw_base(axes[1, 1], r"步骤 3.3：沿 $G_k^c$ 通过 BFS 执行 `expandCluster`")
draw_points(axes[1, 1])
for child, parent in parents.items():
    axes[1, 1].annotate(
        "",
        xy=c_all[child],
        xytext=c_all[parent],
        arrowprops=dict(arrowstyle="->", color="#16A34A", lw=1.7, alpha=0.85),
    )
for node, layer in layers.items():
    edge = layer_edge.get(layer, "#FACC15")
    size = 180 if layer == 0 else 135 if layer == 1 else 115
    axes[1, 1].scatter(*c_all[node], s=size, facecolors="#1F77B4", edgecolors=edge, linewidth=2.8, zorder=8)
    axes[1, 1].text(
        c_all[node][0] + 0.02,
        c_all[node][1] + 0.03,
        rf"$L_{{{layer}}}$",
        fontsize=10.8,
        color="#7C2D12",
        weight="bold",
        zorder=9,
    )
add_caption(
    axes[1, 1],
    r"`expandCluster` 从软核种子出发，对 $G_k^c$ 中可达的点逐步赋予簇标签。" "\n"
    r"只有新到达的软核成员会继续入队，从而驱动后续扩展。",
    "#9A3412",
    "#FFF3E7",
)

draw_base(axes[2, 0], r"步骤 3.4a：按 $n_{Th}$ 过滤前的聚类结果")
draw_points(axes[2, 0], cluster_colors=cluster_palette)
for cid in range(1, cur_id + 1):
    members = np.where(cluster_id == cid)[0]
    if len(members) < 2:
        continue
    center = np.mean(c_all[members], axis=0)
    radius = max(np.linalg.norm(c_all[m] - center) for m in members) + 0.12
    valid = cid in valid_clusters
    axes[2, 0].add_patch(
        Circle(
            center,
            radius,
            facecolor=cluster_palette.get(cid, "#9CA3AF"),
            edgecolor=cluster_palette.get(cid, "#9CA3AF") if valid else "#DC2626",
            linewidth=2.2,
            linestyle="-" if valid else (0, (5, 3)),
            alpha=0.15,
            zorder=0,
        )
    )
    if not valid:
        for m in members:
            axes[2, 0].scatter(*c_all[m], s=102, facecolors="none", edgecolors="#DC2626", linewidth=2.2, zorder=8)
cluster_text = ", ".join([rf"$C_{cid}$: {cluster_sizes[cid]}" for cid in range(1, cur_id + 1)])
add_caption(
    axes[2, 0],
    rf"NBDOS 先根据软核可达性发现 {cur_id} 个候选簇。" "\n"
    rf"随后删除规模小于 $n_{{Th}}$ 的簇；本例中 $n_{{Th}}={nTh}$，各簇规模为 {cluster_text}。",
    "#991B1B",
    "#FDECEC",
)

draw_base(axes[2, 1], r"步骤 3.4b：最终划分为 $O_i^c$ 与 $T_i^c$")
draw_points(axes[2, 1], cluster_colors=cluster_palette, show_trapped_gray=True)
for cid in valid_clusters:
    members = np.where(cluster_id == cid)[0]
    if len(members) < 2:
        continue
    center = np.mean(c_all[members], axis=0)
    radius = max(np.linalg.norm(c_all[m] - center) for m in members) + 0.12
    clr = cluster_palette.get(cid, "#3366CC")
    axes[2, 1].add_patch(
        Circle(center, radius, facecolor=clr, edgecolor=clr, linewidth=2.2, alpha=0.14, zorder=0)
    )
add_caption(
    axes[2, 1],
    rf"有效簇中的成员构成 $O_i^c$，其余 class-$c$ 样本构成 $T_i^c$。" "\n"
    rf"本例中 $|O_i^c|={int(np.sum(is_Oi))}$，$|T_i^c|={int(np.sum(is_Ti))}$。",
    "#7E22CE",
    "#F5ECFC",
)

legend_specs = [
    (r"$x\in S_c$（同类样本）", "o", "#1F77B4", "white", 10, 1.0, "None"),
    (r"$x\in S_{\tilde c}$（异类样本）", "^", "#C65A46", "white", 10, 1.0, "None"),
    (r"软核实例", "o", "#1F77B4", "#16A34A", 11, 1.0, "None"),
    (r"目标点 / 种子点", "o", "#1F77B4", "#F97316", 11, 1.0, "None"),
    (r"簇内连接或扩展路径", "o", "#FFFFFF", "#16A34A", 0.1, 1.0, "-"),
    (r"困境实例 $T_i^c$", "o", "#B9BEC5", "#6B7280", 10, 1.0, "None"),
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
    columnspacing=1.5,
)

fig.suptitle("SMOM NBDOS 聚类过程示意图", fontsize=18, fontweight="bold", y=1.02)

output_path = r"D:\imb_project\Paper\SMOM_NBDOS聚类示意图.png"
ascii_output_path = r"D:\imb_project\Paper\smom_nbdos_clustering_diagram.png"
plt.savefig(ascii_output_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
shutil.copyfile(ascii_output_path, output_path)

print(f"Total class-c instances: {n_c}")
print(f"Soft-core instances: {np.sum(soft_core)} -> {sf_idx.tolist()}")
print(f"Clusters discovered: {cur_id}")
for cid in range(1, cur_id + 1):
    members = np.where(cluster_id == cid)[0].tolist()
    flag = "VALID" if cid in valid_clusters else "DROPPED"
    print(f"  Cluster {cid}: {len(members)} members {members} [{flag}]")
print(f"Outstanding O_i^c: {int(np.sum(is_Oi))}")
print(f"Trapped T_i^c: {int(np.sum(is_Ti))}")
print(f"Saved to: {ascii_output_path}")
print(f"Copied to: {output_path}")
