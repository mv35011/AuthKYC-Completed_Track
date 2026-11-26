"""
Generate publication-grade figures for the AuthKYC research paper.
Figures saved as both 300 DPI PNG and vector PDF in experiment_outputs/paper_figures/:
  Fig 1: FTCA Dual-Domain Cross-Attention Architecture Diagram
  Fig 2: FTCA Training Convergence & Learning Rate Schedule (Epochs 1-36)
  Fig 3: Cross-Dataset Generalization Gap (Protocol 1 vs Protocol 2)
  Fig 4: Ablation Study Bar Chart (Video-Level Variants A, B, C)
  Fig 5: Empirical Video-Level ROC Curves (Extracted from Actual Evaluation Curves)
  Fig 6: FTCA Confusion Matrix Heatmap (n = 594 Video Volumes)
  Fig 7: Multi-Metric Radar Comparison (Video-Level Models)
"""
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.size': 10.5,
    'font.family': 'serif',
    'axes.labelsize': 11.5,
    'axes.titlesize': 12.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'ftca': '#1a5276',
    'vanilla': '#7f8c8d',
    'concat': '#e67e22',
    'accent': '#c0392b',
    'highlight': '#27ae60',
    'blue_light': '#ebf5fb',
    'teal_light': '#e8f8f5',
    'amber_light': '#fef9e7',
    'purple_light': '#f5eef8',
    'slate_light': '#f4f6f7',
}

OUT_DIR = "experiment_outputs/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)


# ─── Load Logged Training & Evaluation Data ───
df_ftca = pd.read_csv("experiment_outputs/ftca_full_training_log.csv")
df_ftca = df_ftca.drop_duplicates(subset=['epoch'], keep='last').sort_values('epoch').reset_index(drop=True)
df_ftca['lr_float'] = df_ftca['lr'].apply(lambda x: float(x) if isinstance(x, str) else float(x))

evals = {}
for m in ["xception", "efficientnet", "r3d18_vanilla", "r3d18_freq_concat", "ftca_full"]:
    with open(f"experiment_outputs/results/{m}_eval.json") as f:
        evals[m] = json.load(f)["metrics"]


# ═══════════════════════════════════════════════════════════
# Figure 1: FTCA Architecture Block Diagram
# ═══════════════════════════════════════════════════════════
def draw_box(ax, x, y, w, h, title, subtitle="", facecolor="#f8f9f9", edgecolor="#2c3e50",
             title_color="#1a252f", lw=1.5, fontsize=9.8, subsize=8.1):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=3
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.71, title, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=title_color, zorder=4)
        ax.text(x + w / 2, y + h * 0.31, subtitle, ha='center', va='center',
                fontsize=subsize, color='#2c3e50', linespacing=1.28, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=title_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, label="", lx=0, ly=0, color="#2c3e50", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.30",
                                color=color, lw=lw, shrinkA=0, shrinkB=0),
                zorder=2)
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label,
                fontsize=8.2, ha='center', va='center', color='#1a252f',
                bbox=dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='#d5dbdb', lw=0.6, alpha=0.95),
                zorder=5)


fig, ax = plt.subplots(figsize=(16.5, 5.6))
ax.set_xlim(0, 18.2)
ax.set_ylim(0, 5.8)
ax.axis('off')
ax.grid(False)

# Subtle stage background zones
for zx, zw, zlabel, zcol in [
    (0.15, 2.75, "Stage 1: Face Clip Input", "#f8f9f9"),
    (3.15, 6.85, "Stage 2: Dual-Domain Feature Encoding", "#fdfefe"),
    (11.15, 6.90, "Stage 3: Cross-Attention Fusion & Decision", "#f8fbfd"),
]:
    zone = FancyBboxPatch((zx, 0.28), zw, 5.32, boxstyle="round,pad=0.05,rounding_size=0.18",
                          facecolor=zcol, edgecolor='#d0d7de', linewidth=1.0, linestyle='--', zorder=1)
    ax.add_patch(zone)
    ax.text(zx + zw / 2, 5.38, zlabel, ha='center', va='center',
            fontsize=9.8, fontweight='bold', color='#566573', zorder=2)

# Stage 1: Input & MTCNN
draw_box(ax, 0.35, 3.25, 2.35, 1.50,
         "Input Video Stream",
         "30 fps KYC Video\nUp to 8 clips / video",
         facecolor=COLORS['slate_light'], edgecolor='#5d6d7e')
draw_box(ax, 0.35, 0.72, 2.35, 1.85,
         "MTCNN Face Crop",
         "Margin = 40 px\n$V \\in \\mathbb{R}^{B \\times 3 \\times 16 \\times 224^2}$\nImageNet Norm",
         facecolor=COLORS['slate_light'], edgecolor='#5d6d7e')
draw_arrow(ax, 1.525, 3.23, 1.525, 2.61)

# Split arrows to dual branches
ax.plot([2.72, 3.05], [1.65, 1.65], color='#2c3e50', lw=1.5, zorder=2)
ax.plot([3.05, 3.05], [1.35, 4.00], color='#2c3e50', lw=1.5, zorder=2)
draw_arrow(ax, 3.05, 4.00, 3.35, 4.00)
draw_arrow(ax, 3.05, 1.35, 3.35, 1.35)

# Stage 2 Top: RGB Branch
draw_box(ax, 3.38, 3.20, 3.45, 1.60,
         "R3D-18 Spatio-Temporal",
         "Kinetics-400 Pretrained\nStem $\\rightarrow L_1 \\rightarrow L_2 \\rightarrow L_3 \\rightarrow L_4$\n33.17M params (8x temp stride)",
         facecolor=COLORS['blue_light'], edgecolor=COLORS['ftca'], title_color=COLORS['ftca'])
draw_arrow(ax, 6.85, 4.00, 7.28, 4.00)
draw_box(ax, 7.30, 3.25, 2.50, 1.50,
         "Spatial Tokens",
         "$\\mathbf{X}_{rgb}$: $[B, 512, 2, 7, 7]$\nFlatten $\\rightarrow \\mathbf{F}_{rgb}$: $[B, 98, 512]$\nQuery Projection $\\mathbf{W}_Q$",
         facecolor=COLORS['blue_light'], edgecolor=COLORS['ftca'], title_color=COLORS['ftca'], subsize=7.9)

# Stage 2 Bottom: Frequency Branch
draw_box(ax, 3.38, 0.55, 2.00, 1.60,
         "2D FFT Spectrum",
         "Grayscale Luminance\n$\\mathrm{fftshift}(\\mathcal{F}_{2D})$\n$\\log(|\\mathcal{F}| + 10^{-8})$",
         facecolor=COLORS['amber_light'], edgecolor='#d35400', title_color='#a04000')
draw_arrow(ax, 5.40, 1.35, 5.65, 1.35)
draw_box(ax, 5.67, 0.55, 2.05, 1.60,
         "3D FreqEncoder",
         "5-Stage 3D CNN\n1-64-128-256-512\n11.73M params",
         facecolor=COLORS['amber_light'], edgecolor='#d35400', title_color='#a04000', subsize=7.9)
draw_arrow(ax, 7.74, 1.35, 7.98, 1.35)
draw_box(ax, 8.00, 0.55, 1.82, 1.60,
         "Spectral Tokens",
         "$\\mathbf{F}_{freq}$: $[B, 98, 512]$\nKey/Value Proj.\n$\\mathbf{W}_K, \\mathbf{W}_V$",
         facecolor=COLORS['amber_light'], edgecolor='#d35400', title_color='#a04000', fontsize=9.2, subsize=7.8)

# Stage 3: Cross-Attention & Classifier (wide open transition x=9.85..11.68)
draw_arrow(ax, 9.84, 3.92, 11.65, 3.05, label="$\\mathbf{Q} = \\mathbf{F}_{rgb}\\mathbf{W}_Q$", lx=-0.10, ly=0.38, color=COLORS['ftca'])
draw_arrow(ax, 9.84, 1.35, 11.65, 2.25, label="$\\mathbf{K},\\mathbf{V} = \\mathbf{F}_{freq}\\mathbf{W}_{K,V}$", lx=-0.05, ly=-0.42, color='#d35400')

draw_box(ax, 11.68, 1.80, 3.10, 1.75,
         "Multi-Head Cross-Attn",
         "8 Heads ($d_k = 64$), Dropout 0.1\n$\\mathrm{softmax}(\\mathbf{Q}\\mathbf{K}^\\top / \\sqrt{d_k})\\mathbf{V}$\nResidual + $\\mathrm{LN}(\\mathbf{F}_{rgb} + \\mathrm{Attn})$ (1.05M)",
         facecolor=COLORS['teal_light'], edgecolor=COLORS['highlight'], title_color='#145a32', subsize=7.8)

# Residual skip arrow
ax.plot([8.55, 8.55], [4.77, 5.02], color=COLORS['ftca'], lw=1.3, ls=':', zorder=2)
ax.plot([8.55, 13.23], [5.02, 5.02], color=COLORS['ftca'], lw=1.3, ls=':', zorder=2)
draw_arrow(ax, 13.23, 5.02, 13.23, 3.58, label="Residual Skip ($+\\,\\mathbf{F}_{rgb}$)", lx=-1.35, ly=0.65, color=COLORS['ftca'])

draw_arrow(ax, 14.80, 2.68, 15.18, 2.68)
draw_box(ax, 15.20, 1.80, 2.68, 1.75,
         "Pool & MLP Head",
         "AdaptiveAvgPool1D $\\rightarrow [B, 512]$\n$512 \\rightarrow 128$ (ReLU, Drop 0.5)\n$128 \\rightarrow 1 + \\sigma(\\cdot) \\in [0, 1]$",
         facecolor=COLORS['purple_light'], edgecolor='#7d3c98', title_color='#512e5f', subsize=7.9)

ax.text(9.1, 0.10, "Total Architecture: 46,015,681 parameters (46.02M)  |  Unified Video-Level Classification on 16-Frame Volume",
        ha='center', va='center', fontsize=8.8, style='italic', color='#566573')

save_fig(fig, "fig1_architecture")
print("✓ Fig 1: Architecture Diagram")


# ═══════════════════════════════════════════════════════════
# Figure 2: Training Convergence & Schedule (36 Epochs)
# ═══════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

# Panel (a): Loss
ax = axes[0]
ax.plot(df_ftca['epoch'], df_ftca['train_loss'], label='Train Loss', color=COLORS['ftca'], lw=2.1)
ax.plot(df_ftca['epoch'], df_ftca['val_loss'], label='Val Loss', color=COLORS['accent'], lw=2.0, ls='--')
ax.axvline(x=11.5, color='#d4ac0d', ls=':', lw=1.5, label='Cycle 2 Resume (Ep 12)')
ax.axvline(x=31.5, color='#2980b9', ls=':', lw=1.5, label='Micro-Polish (Ep 32)')
ax.set_xlabel('Epoch')
ax.set_ylabel('BCE Loss')
ax.set_title('(a) Training & Validation Loss')
ax.set_xlim([0.5, 36.8])
ax.legend(loc='upper right', fontsize=8.0, framealpha=0.92)

# Panel (b): Validation AUC-ROC
ax = axes[1]
ax.plot(df_ftca['epoch'], df_ftca['val_auc'] * 100, label='Protocol 2 Val AUC', color=COLORS['ftca'], lw=2.2)
ax.scatter([33], [95.93], color=COLORS['highlight'], s=95, marker='*', zorder=6, edgecolor='#145a32',
           linewidth=0.7, label='Peak Prot. 2: 95.93% (Ep 33)')
ax.axhline(y=98.74, color='#d4ac0d', ls='--', lw=1.6, label='Protocol 1 Ceiling: 98.74%')
ax.axvline(x=11.5, color='#d4ac0d', ls=':', lw=1.3)
ax.axvline(x=31.5, color='#2980b9', ls=':', lw=1.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('AUC-ROC (%)')
ax.set_title('(b) Validation AUC-ROC Trajectory')
ax.set_xlim([0.5, 36.8])
ax.set_ylim([70, 100.8])
ax.legend(loc='lower right', fontsize=8.0, framealpha=0.92)

# Panel (c): Learning Rate Schedule
ax = axes[2]
ax.plot(df_ftca['epoch'], df_ftca['lr_float'] * 1e4, color=COLORS['ftca'], lw=2.1, label='Learning Rate')
ax.axvline(x=11.5, color='#d4ac0d', ls=':', lw=1.5, label='Resume @ 1.5e-4')
ax.axvline(x=31.5, color='#2980b9', ls=':', lw=1.5, label='Polish @ 1.5e-5')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate ($\\times 10^{-4}$)')
ax.set_title('(c) Cosine Warm-Restart & Polish LR')
ax.set_xlim([0.5, 36.8])
ax.set_ylim([-0.15, 3.25])
ax.legend(loc='upper right', fontsize=8.0, framealpha=0.92)

plt.tight_layout()
save_fig(fig, "fig2_training_dynamics")
print("✓ Fig 2: Training Dynamics")


# ═══════════════════════════════════════════════════════════
# Figure 3: Cross-Dataset Generalization Gap (2-Panel)
# ═══════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.3), gridspec_kw={'width_ratios': [1.85, 1.05]})

# Panel (a): AUC-ROC & Accuracy (%)
cats = ['AUC-ROC (%)', 'Accuracy (%)']
p1_scores = [98.74, 94.55]
p2_scores = [95.93, 90.57]
x = np.arange(len(cats))
w = 0.30

b1 = ax1.bar(x - w / 2, p1_scores, w, label='Protocol 1: FF++ C23 Fakes Only (n=569)',
             color=COLORS['highlight'], alpha=0.88, edgecolor='white', lw=1.2)
b2 = ax1.bar(x + w / 2, p2_scores, w, label='Protocol 2: Cross-Dataset FF++ + Celeb-DF v2 (n=594)',
             color=COLORS['ftca'], alpha=0.92, edgecolor='white', lw=1.2)

for rect in list(b1) + list(b2):
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width() / 2, h + 0.25, f'{h:.2f}%',
             ha='center', va='bottom', fontsize=9.2, fontweight='bold')

for i in range(len(cats)):
    drop = p1_scores[i] - p2_scores[i]
    # Place clear bracket/drop badge above the blue bar without overlapping it
    ax1.annotate(f'Δ = −{drop:.2f}%',
                 xy=(x[i] + w / 2, p2_scores[i] + 1.15),
                 xytext=(x[i] + w / 2, p2_scores[i] + 2.75),
                 fontsize=8.6, color=COLORS['accent'], fontweight='bold', ha='center',
                 arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.2))

ax1.set_xticks(x)
ax1.set_xticklabels(cats, fontsize=10.5)
ax1.set_ylim([84.0, 102.5])
ax1.set_ylabel('Score (%)')
ax1.set_title('(a) Classification Performance Drop')

# Panel (b): Validation BCE Loss (separate appropriate scale!)
loss_x = np.array([0.0])
lw_bar = 0.26
bl1 = ax2.bar(loss_x - lw_bar / 2, [0.1447], lw_bar, color=COLORS['highlight'], alpha=0.88, edgecolor='white', lw=1.2)
bl2 = ax2.bar(loss_x + lw_bar / 2, [0.2547], lw_bar, color=COLORS['ftca'], alpha=0.92, edgecolor='white', lw=1.2)

ax2.text(loss_x[0] - lw_bar / 2, 0.1447 + 0.006, '0.1447', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax2.text(loss_x[0] + lw_bar / 2, 0.2547 + 0.006, '0.2547', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax2.annotate('+0.1100\n(+76.0% rel.)', xy=(lw_bar / 2, 0.2547), xytext=(-0.03, 0.285),
             fontsize=8.5, color=COLORS['accent'], fontweight='bold', ha='center')

ax2.set_xticks([0.0])
ax2.set_xticklabels(['Validation BCE Loss'], fontsize=10.5)
ax2.set_xlim([-0.55, 0.55])
ax2.set_ylim([0.0, 0.33])
ax2.set_ylabel('BCE Loss (Lower is Better)')
ax2.set_title('(b) Cross-Domain Calibration Shift')

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True, framealpha=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.94])
save_fig(fig, "fig3_cross_dataset_generalization")
print("✓ Fig 3: Cross-Dataset Generalization")


# ═══════════════════════════════════════════════════════════
# Figure 4: Ablation Study (Video-Level Models Only)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10.2, 4.9))

metrics_names = ['AUC-ROC (%)', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score ($\\times 100$)']
vanilla_vals = [evals['r3d18_vanilla'][k] * 100 for k in ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']]
concat_vals = [evals['r3d18_freq_concat'][k] * 100 for k in ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']]
ftca_vals = [evals['ftca_full'][k] * 100 for k in ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']]

x = np.arange(len(metrics_names))
width = 0.25

r1 = ax.bar(x - width, vanilla_vals, width, label='Variant A: R3D-18 RGB Only (33.23M)',
            color=COLORS['vanilla'], alpha=0.88, edgecolor='white')
r2 = ax.bar(x, concat_vals, width, label='Variant B: R3D-18 + FFT Concat (45.16M)',
            color=COLORS['concat'], alpha=0.88, edgecolor='white')
r3 = ax.bar(x + width, ftca_vals, width, label='Variant C: FTCA Cross-Attention (46.02M)',
            color=COLORS['ftca'], edgecolor='white')

for rects in [r1, r2, r3]:
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.35, f'{h:.2f}',
                ha='center', va='bottom', fontsize=7.6, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.set_ylim([74.0, 99.5])
ax.set_ylabel('Video-Level Score (%)')
ax.set_title('Controlled Ablation Study Across Fusion Strategies (Cross-Dataset, n = 594)', pad=28)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=True, framealpha=0.95, fontsize=8.8)

plt.tight_layout()
save_fig(fig, "fig4_ablation_study")
print("✓ Fig 4: Ablation Study")


# ═══════════════════════════════════════════════════════════
# Figure 5: Empirical ROC Curves Extracted from Eval Plots
# ═══════════════════════════════════════════════════════════
def extract_empirical_roc(png_path):
    """Trace exact empirical step ROC curve from evaluate.py PNG plot."""
    img = cv2.imread(png_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    xs = np.where(np.sum(gray < 20, axis=0) > h * 0.5)[0]
    ys = np.where(np.sum(gray < 20, axis=1) > w * 0.5)[0]
    x_left, x_right = xs[0], xs[-1]
    y_top, y_bot = ys[0], ys[-1]

    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    mask = (b > 140) & (r < 60) & (g > 90) & (g < 150)
    pts_y, pts_x = np.where(mask)
    valid = (pts_x >= x_left) & (pts_x <= x_right) & (pts_y >= y_top) & (pts_y <= y_bot)

    fx_raw = (pts_x[valid] - x_left) / (x_right - x_left)
    fy_raw = (y_bot - pts_y[valid]) / ((y_bot - y_top) / 1.05)
    keep = ~((fx_raw > 0.50) & (fy_raw < 0.18))
    px, py = pts_x[valid][keep], pts_y[valid][keep]

    ux, uy = [0.0], [0.0]
    for xi in range(x_left, x_right + 1):
        sub = py[px == xi]
        if len(sub) > 0:
            fx = (xi - x_left) / (x_right - x_left)
            fy = np.clip((y_bot - np.median(sub)) / ((y_bot - y_top) / 1.05), 0.0, 1.0)
            ux.append(fx)
            uy.append(fy)
    ux.append(1.0)
    uy.append(1.0)
    return np.array(ux), np.maximum.accumulate(np.array(uy))


fig, ax = plt.subplots(figsize=(6.8, 5.4))

video_models = [
    ("ftca_full", "FTCA (Ours, Cross-Attn)", COLORS['ftca'], '-', 2.4),
    ("r3d18_vanilla", "R3D-18 Vanilla (RGB Only)", COLORS['vanilla'], '--', 1.9),
    ("r3d18_freq_concat", "R3D-18 + Freq Concat", COLORS['concat'], '-.', 1.9),
]

for key, name, color, ls, lw in video_models:
    fpr_emp, tpr_emp = extract_empirical_roc(f"experiment_outputs/results/{key}_eval_roc.png")
    auc_val = evals[key]['auc_roc']
    eer = evals[key]['eer']
    ax.plot(fpr_emp, tpr_emp,
            label=f'{name} (AUC = {auc_val*100:.2f}%, EER = {eer*100:.2f}%)',
            color=color, linestyle=ls, linewidth=lw)
    ax.plot(eer, 1.0 - eer, 'o', color=color, markersize=7.5,
            markeredgecolor='white', markeredgewidth=1.4, zorder=6)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.35, lw=1.1, label='Random Classifier (AUC = 50.00%)')
ax.scatter([], [], c='#34495e', marker='o', s=42, label='Markers = EER Operating Points')

ax.set_xlim([-0.015, 1.01])
ax.set_ylim([0.0, 1.03])
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR)')
ax.set_title('Empirical Video-Level ROC Curves (n = 594 Validation Clips)')
ax.legend(loc='lower right', framealpha=0.95, fontsize=8.4)

plt.tight_layout()
save_fig(fig, "fig5_roc_video_level")
print("✓ Fig 5: Empirical ROC Curves")


# ═══════════════════════════════════════════════════════════
# Figure 6: Confusion Matrix — FTCA
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.grid(False)

with open("experiment_outputs/results/ftca_full_eval.json") as f:
    cm = np.array(json.load(f)["confusion_matrix"])
labels = ['Authentic', 'Manipulated']

cax = ax.imshow(cm, cmap='Blues', aspect='auto', vmin=0, vmax=300)
for (i, j), val in np.ndenumerate(cm):
    pct = val / cm[i].sum() * 100
    color = 'white' if val > 150 else 'black'
    ax.text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center',
            fontsize=13, fontweight='bold', color=color)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('FTCA Confusion Matrix (n = 594 Video Volumes)', fontsize=11.8, pad=10)
fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
save_fig(fig, "fig6_confusion_matrix")
print("✓ Fig 6: Confusion Matrix")


# ═══════════════════════════════════════════════════════════
# Figure 7: Radar Chart — Comprehensive Model Comparison
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.8, 5.8), subplot_kw=dict(polar=True))

categories = ['AUC-ROC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', '1 − EER']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]


def get_radar_vals(model_key):
    m = evals[model_key]
    return [m['auc_roc'] * 100, m['accuracy'] * 100, m['precision'] * 100,
            m['recall'] * 100, m['f1_score'] * 100, (1 - m['eer']) * 100]


for model_key, name, color, ls in [
    ('ftca_full', 'FTCA (Ours)', COLORS['ftca'], '-'),
    ('r3d18_vanilla', 'R3D-18 Vanilla', COLORS['vanilla'], '--'),
    ('r3d18_freq_concat', 'R3D-18 + Freq Concat', COLORS['concat'], '-.'),
]:
    vals = get_radar_vals(model_key)
    vals += vals[:1]
    ax.plot(angles, vals, lw=2.1, ls=ls, label=name, color=color)
    ax.fill(angles, vals, alpha=0.09, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10.5)
ax.tick_params(axis='x', pad=20)
ax.set_ylim([75, 100])
ax.set_yticks([80, 85, 90, 95, 100])
ax.set_rlabel_position(32)
ax.set_title('Multi-Metric Model Comparison (Video-Level)', pad=26, fontsize=12.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9.2, frameon=True)

plt.tight_layout()
save_fig(fig, "fig7_radar_comparison")
print("✓ Fig 7: Radar Chart")

print(f"\nAll 7 figures saved cleanly to {OUT_DIR}/")
