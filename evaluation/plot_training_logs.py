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


def plot_metrics(data, output_png, title="Training & Evaluation Metrics"):
    """2x2 grid; train and eval losses get separate panels because their scales differ
    (train = CE + triplet on raw features, eval = triplet on cosine distances)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_loss, ax_val_loss = axes[0]
    ax_sub, ax_val = axes[1]
    eval_epochs = data.get("eval_epochs", [])

    # --- Panel 1: Training loss + learning rate ---
    ax_loss.plot(data["epochs"], data["losses"], color="tab:red", marker="o", markersize=3,
                 label="Train Loss", linewidth=1.8)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Train Loss", fontsize=11, color="tab:red")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    ax_loss.set_title("Training Loss", fontsize=13, fontweight="bold")
    if any(data.get("lrs", [])):
        ax_lr = ax_loss.twinx()
        ax_lr.plot(data["epochs"], data["lrs"], color="tab:gray", linestyle=":", alpha=0.8, label="LR")
        ax_lr.set_ylabel("Learning Rate (log)", color="tab:gray", fontsize=11)
        ax_lr.tick_params(axis="y", labelcolor="tab:gray")
        ax_lr.set_yscale("log")
        h1, l1 = ax_loss.get_legend_handles_labels()
        h2, l2 = ax_lr.get_legend_handles_labels()
        ax_loss.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)
    else:
        ax_loss.legend(loc="upper right", frameon=True)

    # --- Panel 2: Evaluation loss + positive/negative distances ---
    val_loss_series = data.get("val_losses", [])
    if val_loss_series and len(val_loss_series) == len(eval_epochs):
        ax_val_loss.plot(eval_epochs, val_loss_series, color="tab:blue", marker="s", markersize=4,
                         label="Eval Loss (hard triplet, cosine)", linewidth=1.8)
        val_pos = data.get("val_pos_dists", [])
        if val_pos and len(val_pos) == len(eval_epochs) and val_pos != val_loss_series:
            ax_val_loss.plot(eval_epochs, val_pos, color="tab:cyan", marker="^", markersize=3,
                             linestyle=":", label="Mean Pos Dist", linewidth=1.4)
        ax_val_loss.legend(loc="best", frameon=True)
    else:
        ax_val_loss.text(0.5, 0.5, "No evaluation yet", ha="center", va="center", transform=ax_val_loss.transAxes)
    ax_val_loss.set_xlabel("Epoch", fontsize=11)
    ax_val_loss.set_ylabel("Eval Loss / Distance", fontsize=11)
    ax_val_loss.set_title("Evaluation Loss", fontsize=13, fontweight="bold")
    ax_val_loss.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 3: Loss decomposition (separate y-axes: CE and triplet differ in scale) ---
    if data.get("cls_losses"):
        ax_sub.plot(data["sub_epochs"], data["cls_losses"], color="tab:purple", marker="v", markersize=3,
                    linewidth=1.8, label="CrossEntropy ($L_{cls}$)")
        ax_sub.set_ylabel("$L_{cls}$", color="tab:purple", fontsize=11)
        ax_tri = ax_sub.twinx()
        ax_tri.plot(data["sub_epochs"], data["tri_losses"], color="tab:orange", marker="^", markersize=3,
                    linewidth=1.8, label="Triplet ($L_{tri}$)")
        ax_tri.set_ylabel("$L_{tri}$", color="tab:orange", fontsize=11)
        h1, l1 = ax_sub.get_legend_handles_labels()
        h2, l2 = ax_tri.get_legend_handles_labels()
        ax_sub.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)
    else:
        ax_sub.text(0.5, 0.5, "No loss components", ha="center", va="center", transform=ax_sub.transAxes)
    ax_sub.set_xlabel("Epoch", fontsize=11)
    ax_sub.set_title("Training Loss Decomposition", fontsize=13, fontweight="bold")
    ax_sub.grid(True, linestyle="--", alpha=0.5)

    # --- Panel 4: Retrieval metrics ---
    if eval_epochs:
        ax_val.plot(eval_epochs, data["rank1"], color="tab:blue", marker="s", markersize=5, linewidth=2, label="Rank-1 (%)")
        ax_val.plot(eval_epochs, data["rank5"], color="tab:cyan", marker="^", markersize=5, linewidth=2, label="Rank-5 (%)")
        ax_val.plot(eval_epochs, data["mAP"], color="tab:green", marker="D", markersize=5, linewidth=2, label="mAP (%)")
        if data["mAP"]:
            best = max(data["mAP"])
            best_ep = eval_epochs[data["mAP"].index(best)]
            ax_val.plot([best_ep], [best], marker="o", color="red", markersize=9)
            ax_val.annotate(f"Best mAP: {best:.1f}% (Ep {best_ep})\nLast mAP: {data['mAP'][-1]:.1f}% (Ep {eval_epochs[-1]})",
                            xy=(best_ep, best), xytext=(0.02, 0.97), textcoords="axes fraction", va="top",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), fontsize=9)
        ax_val.legend(loc="lower right", frameon=True, shadow=True)
    ax_val.set_xlabel("Epoch", fontsize=11)
    ax_val.set_ylabel("Retrieval Score (%)", fontsize=11)
    ax_val.set_title("Evaluation Metrics", fontsize=13, fontweight="bold")
    ax_val.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"-> [SUCCESS] Saved plot to {output_png}")

    # Also save standalone separate plots so train and val loss can be viewed in isolation
    out_dir = out_path.parent
    _plot_train_loss(data, out_dir / "loss_train.png", title=f"Training Loss ({out_dir.name})")
    _plot_val_loss(data, out_dir / "loss_val.png", title=f"Evaluation Loss ({out_dir.name})")
    _plot_eval_metrics(data, out_dir / "eval_metrics.png", title=f"Retrieval Metrics ({out_dir.name})")


def _plot_train_loss(data, output_png, title="Training Loss"):
    if not data.get("epochs"):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["epochs"], data["losses"], color="tab:red", marker="o", markersize=3,
            label="Train Loss", linewidth=1.8)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Train Loss", fontsize=11, color="tab:red")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if any(data.get("lrs", [])):
        ax_lr = ax.twinx()
        ax_lr.plot(data["epochs"], data["lrs"], color="tab:gray", linestyle=":", alpha=0.8, label="LR")
        ax_lr.set_ylabel("Learning Rate (log)", color="tab:gray", fontsize=11)
        ax_lr.set_yscale("log")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_lr.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)
    else:
        ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_val_loss(data, output_png, title="Evaluation Loss"):
    eval_epochs = data.get("eval_epochs", [])
    val_loss_series = data.get("val_losses", [])
    if not eval_epochs or not val_loss_series:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_epochs, val_loss_series, color="tab:blue", marker="s", markersize=4,
            label="Eval Loss (hard triplet, cosine)", linewidth=1.8)
    val_pos = data.get("val_pos_dists", [])
    if val_pos and len(val_pos) == len(eval_epochs) and val_pos != val_loss_series:
        ax.plot(eval_epochs, val_pos, color="tab:cyan", marker="^", markersize=3,
                linestyle=":", label="Mean Pos Dist", linewidth=1.4)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Eval Loss / Distance", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_eval_metrics(data, output_png, title="Retrieval Metrics"):
    eval_epochs = data.get("eval_epochs", [])
    if not eval_epochs or not data.get("rank1"):
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eval_epochs, data["rank1"], color="tab:blue", marker="s", markersize=5, linewidth=2, label="Rank-1 (%)")
    ax.plot(eval_epochs, data["rank5"], color="tab:cyan", marker="^", markersize=5, linewidth=2, label="Rank-5 (%)")
    ax.plot(eval_epochs, data["mAP"], color="tab:green", marker="D", markersize=5, linewidth=2, label="mAP (%)")
    if data.get("mAP"):
        best = max(data["mAP"])
        best_ep = eval_epochs[data["mAP"].index(best)]
        ax.plot([best_ep], [best], marker="o", color="red", markersize=9)
        ax.annotate(f"Best mAP: {best:.1f}% (Ep {best_ep})\nLast mAP: {data['mAP'][-1]:.1f}% (Ep {eval_epochs[-1]})",
                    xy=(best_ep, best), xytext=(0.02, 0.95), textcoords="axes fraction", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7), fontsize=9)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Retrieval Score (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()


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
