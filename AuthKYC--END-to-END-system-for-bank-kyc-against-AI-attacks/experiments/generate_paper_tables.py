"""
Generate LaTeX and Markdown comparison tables for AuthKYC paper.
Reads all evaluation JSONs from experiment_outputs/results/ and outputs:
- Table 1: Model Benchmark Comparison
- Table 2: Ablation Study
"""
import os
import json
import glob

def main():
    results_dir = "experiment_outputs/results"
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} not found. Run evaluations first.")
        return

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print(f"No evaluation JSON files found in {results_dir}.")
        return

    data = {}
    for f in json_files:
        with open(f, 'r') as fp:
            res = json.load(fp)
            model_name = res.get("model", os.path.basename(f).replace("_eval.json", ""))
            data[model_name] = res["metrics"]

    # Param counts
    params = {
        "xception": "22.8M",
        "efficientnet": "19.3M",
        "r3d18_vanilla": "33.2M",
        "r3d18_freq_concat": "34.5M",
        "ftca_full": "46.0M"
    }

    display_names = {
        "xception": "XceptionNet (Rössler et al.)",
        "efficientnet": "EfficientNet-B4",
        "r3d18_vanilla": "R3D-18 (Vanilla RGB)",
        "r3d18_freq_concat": "R3D-18 + Freq (Concat)",
        "ftca_full": "FTCA (Ours — Cross-Attention)"
    }

    types = {
        "xception": "2D Frame (299x299)",
        "efficientnet": "2D Frame (300x300)",
        "r3d18_vanilla": "3D Clip (16x224x224)",
        "r3d18_freq_concat": "3D Clip + FFT",
        "ftca_full": "3D Clip + FFT"
    }

    print("\n" + "="*80)
    print("TABLE 1: MODEL BENCHMARK COMPARISON (MARKDOWN)")
    print("="*80)
    print("| Method | Modality | Params | Accuracy (%) | AUC-ROC (%) | Precision (%) | Recall (%) | F1-Score | EER (%) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for k in ["xception", "efficientnet", "r3d18_vanilla", "r3d18_freq_concat", "ftca_full"]:
        if k in data:
            m = data[k]
            acc = m.get("accuracy", 0) * 100
            auc = m.get("auc_roc", 0) * 100
            prec = m.get("precision", 0) * 100
            rec = m.get("recall", 0) * 100
            f1 = m.get("f1_score", 0)
            eer = m.get("eer", 0) * 100
            bold = "**" if k == "ftca_full" else ""
            print(f"| {bold}{display_names[k]}{bold} | {types[k]} | {params[k]} | {bold}{acc:.2f}%{bold} | {bold}{auc:.2f}%{bold} | {prec:.2f}% | {rec:.2f}% | {f1:.4f} | {eer:.2f}% |")

    print("\n" + "="*80)
    print("TABLE 2: ABLATION STUDY (MARKDOWN)")
    print("="*80)
    print("| Architecture Variant | Spatio-Temporal RGB | FFT Freq Stream | Fusion Strategy | AUC-ROC (%) | EER (%) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    ablations = [
        ("r3d18_vanilla", "R3D-18 Backbone Only", "✓", "✗", "None"),
        ("r3d18_freq_concat", "R3D-18 + Freq (Concat)", "✓", "✓", "Feature Concatenation"),
        ("ftca_full", "Full FTCA (Dual-Domain)", "✓", "✓", "Cross-Attention ($Q_{rgb}, K_{freq}, V_{freq}$)")
    ]

    for key, name, rgb, freq, fusion in ablations:
        if key in data:
            m = data[key]
            auc = m.get("auc_roc", 0) * 100
            eer = m.get("eer", 0) * 100
            bold = "**" if key == "ftca_full" else ""
            print(f"| {bold}{name}{bold} | {rgb} | {freq} | {fusion} | {bold}{auc:.2f}%{bold} | {eer:.2f}% |")

    # LaTeX Output
    print("\n" + "="*80)
    print("TABLE 1 (LATEX FORMAT FOR PAPER)")
    print("="*80)
    print(r"""\begin{table*}[t]
\centering
\caption{Comparative performance of AuthKYC (FTCA) against frame-level and spatio-temporal baselines on FaceForensics++ (C23) and Celeb-DF v2.}
\label{tab:benchmark}
\begin{tabular}{llccccccc}
\toprule
\textbf{Method} & \textbf{Modality} & \textbf{Params} & \textbf{Accuracy (\%)} & \textbf{AUC-ROC (\%)} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1} & \textbf{EER (\%)} \\
\midrule""")
    for k in ["xception", "efficientnet", "r3d18_vanilla", "r3d18_freq_concat", "ftca_full"]:
        if k in data:
            m = data[k]
            acc = m.get("accuracy", 0) * 100
            auc = m.get("auc_roc", 0) * 100
            prec = m.get("precision", 0) * 100
            rec = m.get("recall", 0) * 100
            f1 = m.get("f1_score", 0)
            eer = m.get("eer", 0) * 100
            if k == "ftca_full":
                print(f"\\textbf{{{display_names[k]}}} & {types[k]} & {params[k]} & \\textbf{{{acc:.2f}}} & \\textbf{{{auc:.2f}}} & \\textbf{{{prec:.2f}}} & \\textbf{{{rec:.2f}}} & \\textbf{{{f1:.4f}}} & \\textbf{{{eer:.2f}}} \\\\")
            else:
                print(f"{display_names[k]} & {types[k]} & {params[k]} & {acc:.2f} & {auc:.2f} & {prec:.2f} & {rec:.2f} & {f1:.4f} & {eer:.2f} \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table*}""")

if __name__ == "__main__":
    main()
