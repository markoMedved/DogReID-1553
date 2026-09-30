import argparse
import re
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless cluster
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_log_files(log_path):
    log_path = Path(log_path)
    epochs, losses, lrs = [], [], []
    eval_epochs, val_losses, val_pos_dists, rank1s, rank5s, maps = [], [], [], [], [], []

    # Check if input is a JSON history file
    if log_path.suffix == ".json":
        with open(log_path, "r") as f:
            data = json.load(f)
        return data

    # Parse .out file
    current_epoch = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            loss_match = re.search(r"Epoch\s+(\d+)\s+\|\s+(?:Train\s+)?Loss:\s+([\d\.]+)\s+\|\s+LR:\s+([^\s]+)", line)
            if loss_match:
                ep = int(loss_match.group(1))
                loss = float(loss_match.group(2))
                lr_str = loss_match.group(3).split("/")[0].strip()
                try:
                    lr = float(lr_str)
                except ValueError:
                    lr = 0.0
                epochs.append(ep)
                losses.append(loss)
                lrs.append(lr)
                current_epoch = ep

            # Format 1: New standardized format:
            # Eval (Closed) -> Val Loss: 0.3521 | Rank-1: 10.44%, Rank-5: 22.15%, mAP: 9.33% | PosDist: 0.5605, NegDist: ...
            # Format 2: Previous format:
            # Eval (Closed) -> Rank-1: 10.44%, Rank-5: 22.15%, mAP: 9.33% | Val PosDist: 0.5605
            eval_match = re.search(r"Eval\s+\(Closed\)\s+->.*?(?:Val Loss:\s+([\d\.]+).*?)?Rank-1:\s+([\d\.]+)%,\s+Rank-5:\s+([\d\.]+)%,\s+mAP:\s+([\d\.]+)%(?:.*?PosDist:\s+([\d\.]+))?", line)
            if eval_match:
                val_l_str = eval_match.group(1)
                r1 = float(eval_match.group(2))
                r5 = float(eval_match.group(3))
                map_val = float(eval_match.group(4))
                pos_d_str = eval_match.group(5)

                val_l = float(val_l_str) if val_l_str else (float(pos_d_str) if pos_d_str else None)
                pos_d = float(pos_d_str) if pos_d_str else None

                target_epoch = current_epoch if current_epoch is not None else (eval_epochs[-1] + 1 if eval_epochs else 0)
                eval_epochs.append(target_epoch)
                rank1s.append(r1)
                rank5s.append(r5)
                maps.append(map_val)
                if val_l is not None:
                    val_losses.append(val_l)
                if pos_d is not None:
                    val_pos_dists.append(pos_d)

    # Attempt to parse sub-losses from .err file
    err_path = log_path.with_suffix(".err")
    cls_losses, tri_losses, sub_epochs = [], [], []
    if err_path.exists():
        epoch_sub = {}
        with open(err_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                matches = re.findall(r"Epoch\s+(\d+):.*?\[cls:([\d\.]+)\s+triplet:([\d\.]+)\]", line)
                for ep, cls_l, tri_l in matches:
                    ep = int(ep)
                    if ep not in epoch_sub:
                        epoch_sub[ep] = {"cls": [], "tri": []}
                    epoch_sub[ep]["cls"].append(float(cls_l))
                    epoch_sub[ep]["tri"].append(float(tri_l))

        for ep in sorted(epoch_sub.keys()):
            if epoch_sub[ep]["cls"] and epoch_sub[ep]["tri"]:
                sub_epochs.append(ep)
                cls_losses.append(sum(epoch_sub[ep]["cls"]) / len(epoch_sub[ep]["cls"]))
                tri_losses.append(sum(epoch_sub[ep]["tri"]) / len(epoch_sub[ep]["tri"]))

    return {
        "epochs": epochs,
        "losses": losses,
        "lrs": lrs,
        "sub_epochs": sub_epochs,
        "cls_losses": cls_losses,
        "tri_losses": tri_losses,
        "eval_epochs": eval_epochs,
        "val_losses": val_losses,
        "val_pos_dists": val_pos_dists,
        "rank1": rank1s,
        "rank5": rank5s,
        "mAP": maps
    }


def plot_metrics(data, output_png, title="Training & Validation Metrics"):
    has_sublosses = bool(data.get("cls_losses"))

    fig, axes = plt.subplots(1, 3 if has_sublosses else 2, figsize=(18 if has_sublosses else 14, 5.2))

    if has_sublosses:
        ax_loss, ax_sub, ax_val = axes[0], axes[1], axes[2]
    else:
        ax_loss, ax_val = axes[0], axes[1]
        ax_sub = None

    # --- Plot 1: Total Training Loss vs Validation Loss & Learning Rate ---
    color = "tab:red"
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.plot(data["epochs"], data["losses"], color=color, marker="o", markersize=3, label="Train Loss", linewidth=1.8)

    # Plot validation loss if available
    val_loss_series = data.get("val_losses", [])
    eval_epochs = data.get("eval_epochs", [])
    if val_loss_series and len(val_loss_series) == len(eval_epochs):
        ax_loss.plot(eval_epochs, val_loss_series, color="tab:blue", marker="s", markersize=4, linestyle="--", label="Val Loss", linewidth=1.8)

    # Optionally plot val pos distance if present
    val_pos_dists = data.get("val_pos_dists", [])
    if val_pos_dists and len(val_pos_dists) == len(eval_epochs) and val_pos_dists != val_loss_series:
        ax_loss.plot(eval_epochs, val_pos_dists, color="tab:cyan", marker="^", markersize=3, linestyle=":", label="Val Pos Dist", linewidth=1.4)

    ax_loss.legend(loc="upper right", frameon=True)
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    if any(data.get("lrs", [])):
        ax_lr = ax_loss.twinx()
        color_lr = "tab:gray"
        ax_lr.set_ylabel("Learning Rate (log)", color=color_lr, fontsize=11)
        ax_lr.plot(data["epochs"], data["lrs"], color=color_lr, linestyle=":", alpha=0.7, label="LR")
        ax_lr.tick_params(axis="y", labelcolor=color_lr)
        ax_lr.set_yscale("log")

    ax_loss.set_title("Train Loss vs Validation Loss", fontsize=13, fontweight="bold")

    # --- Plot 2: Loss Decomposition (Classification vs Triplet) ---
    if ax_sub and has_sublosses:
        ax_sub.plot(data["sub_epochs"], data["cls_losses"], color="tab:purple", marker="v", markersize=3, linewidth=1.8, label="CrossEntropy ($L_{cls}$)")
        ax_sub.plot(data["sub_epochs"], data["tri_losses"], color="tab:orange", marker="^", markersize=3, linewidth=1.8, label="Triplet ($L_{tri}$)")
        ax_sub.set_xlabel("Epoch", fontsize=11)
        ax_sub.set_ylabel("Component Loss", fontsize=11)
        ax_sub.set_title("Loss Decomposition", fontsize=13, fontweight="bold")
        ax_sub.legend(loc="best", frameon=True)
        ax_sub.grid(True, linestyle="--", alpha=0.5)

    # --- Plot 3: Validation Retrieval Metrics (Rank-1, Rank-5, mAP) ---
    if data.get("eval_epochs"):
        ax_val.plot(data["eval_epochs"], data["rank1"], color="tab:blue", marker="s", markersize=5, linewidth=2, label="Rank-1 (%)")
        ax_val.plot(data["eval_epochs"], data["rank5"], color="tab:cyan", marker="^", markersize=5, linewidth=2, label="Rank-5 (%)")
        ax_val.plot(data["eval_epochs"], data["mAP"], color="tab:green", marker="D", markersize=5, linewidth=2, label="mAP (%)")

        if data["rank1"]:
            max_r1 = max(data["rank1"])
            max_idx = data["rank1"].index(max_r1)
            best_epoch = data["eval_epochs"][max_idx]
            ax_val.plot([best_epoch], [max_r1], marker="o", color="red", markersize=9)
            ax_val.annotate(
                f"Best: {max_r1:.1f}% (Ep {best_epoch})",
                xy=(best_epoch, max_r1),
                xytext=(best_epoch, max_r1 + 0.5),
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7),
                fontsize=9
            )

        ax_val.set_xlabel("Epoch", fontsize=11)
        ax_val.set_ylabel("Retrieval Score (%)", fontsize=11)
        ax_val.set_title("Validation Metrics (Unseen Dogs)", fontsize=13, fontweight="bold")
        ax_val.legend(loc="upper right", frameon=True, shadow=True)
        ax_val.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> [SUCCESS] Saved plot to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Parse training logs / history JSON and plot publication curves.")
    parser.add_argument("log_file", type=str, help="Path to .out, .err, or training_history.json")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image file path (default: <log_stem>_curves.png)")
    parser.add_argument("--title", "-t", type=str, default=None, help="Plot figure title")
    args = parser.parse_args()

    input_path = Path(args.log_file)
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    data = parse_log_files(input_path)

    out_file = args.output
    if out_file is None:
        if input_path.suffix == ".json":
            out_file = str(input_path.with_suffix(".png"))
        else:
            out_file = str(input_path.parent / f"{input_path.stem}_curves.png")

    title = args.title or f"Metrics: {input_path.stem}"
    num_ep = len(data.get("epochs", []))
    num_sub = len(data.get("sub_epochs", []))
    num_eval = len(data.get("eval_epochs", []))
    print(f"Parsed {num_ep} training epochs, {num_sub} subloss epochs, and {num_eval} validation evaluations.")

    plot_metrics(data, out_file, title=title)


if __name__ == "__main__":
    main()
