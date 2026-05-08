"""Plot validation AP50 over epochs for custom vs. default anchors.

Reads the two history.json files (custom: ./output, default: ./
output_ablation_default_anchors) and saves a side-by-side comparison
to ./output/exp_anchor_ablation.png — used in the §4.1 ablation.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt


def _load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom", default="./output/history.json")
    parser.add_argument("--default", default="./output_ablation_default_anchors/history.json")
    parser.add_argument("--out", default="./output/exp_anchor_ablation.png")
    args = parser.parse_args()

    custom = _load(args.custom)
    default = _load(args.default)

    def series(history):
        eps = [r["epoch"] for r in history if "val_AP50" in r]
        return eps, [r["val_AP50"] for r in history if "val_AP50" in r], \
               [r["val_AP"]   for r in history if "val_AP50" in r]

    c_e, c_ap50, c_ap = series(custom)
    d_e, d_ap50, d_ap = series(default)

    n_match = min(max(c_e), max(d_e))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(c_e, c_ap50, marker="o", color="tab:blue",
            label="custom anchors {8,16,32,64,128} — final model")
    ax.plot(d_e, d_ap50, marker="s", color="tab:red",
            label="default anchors {32,64,128,256,512} — ablation")
    ax.axhline(0.5142, color="tab:blue", linestyle="--", alpha=0.4,
               label=f"custom peak val AP50 = 0.5142 (epoch 11)")
    ax.axhline(max(d_ap50), color="tab:red", linestyle="--", alpha=0.4,
               label=f"default peak val AP50 = {max(d_ap50):.4f} (epoch {d_e[d_ap50.index(max(d_ap50))]})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AP50")
    ax.set_title("E1 ablation: validation AP50 — custom vs. default anchors\n"
                 "(identical training pipeline, both trained from scratch)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)

    print("Comparison summary (val AP50):")
    print(f"  custom anchors  : peak {max(c_ap50):.4f} at epoch {c_e[c_ap50.index(max(c_ap50))]} "
          f"(history has {len(c_ap50)} eval points up to epoch {max(c_e)})")
    print(f"  default anchors : peak {max(d_ap50):.4f} at epoch {d_e[d_ap50.index(max(d_ap50))]} "
          f"(history has {len(d_ap50)} eval points up to epoch {max(d_e)})")
    print(f"  gap at peak    : {max(c_ap50) - max(d_ap50):+.4f}")
    print(f"\nFirst {min(len(c_ap50), len(d_ap50))} eval points compared:")
    for i in range(min(len(c_ap50), len(d_ap50))):
        e_c = c_e[i]; e_d = d_e[i]
        print(f"  epoch {e_c:>2}: custom={c_ap50[i]:.4f}  default={d_ap50[i]:.4f}  "
              f"Δ={c_ap50[i]-d_ap50[i]:+.4f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
