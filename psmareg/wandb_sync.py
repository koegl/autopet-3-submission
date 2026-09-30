"""Sync nnU-Net training logs of all folds to Weights & Biases (lightweight, meant for the login node).

One wandb run per fold with a fixed id, so restarting this script or the 24h training jobs resumes the same run.
Each poll re-parses all training_log_*.txt of a fold (a resumed training job writes a new file; epochs repeated
after a resume are taken from the newest file) and logs every completed epoch not yet in wandb, using the epoch as
step. Epochs that were already synced and then re-run after a resume are not overwritten.

Usage:  python psmareg/wandb_sync.py                      # poll every 10 min until all folds are done
        python psmareg/wandb_sync.py --once --folds 0 1   # single sync
"""
import argparse
import json
import re
import time
from pathlib import Path

import wandb

RESULTS = Path("/lustre/groups/iml/data/PSMAReg/lesion_seg_nnunet_autopet3/nnUNet_results/Dataset901_PSMAReg_lesions/"
               "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres")

LINE = re.compile(r"^\d{4}-\d{2}-\d{2} [\d:.]+: (.*?)\s*$")
FLOAT = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?")


def log_files(fold_dir):
    """training_log_<Y>_<m>_<d>_<H>_<M>_<S>.txt, sorted chronologically (names are not zero-padded)."""
    return sorted(fold_dir.glob("training_log_*.txt"), key=lambda p: [int(x) for x in p.stem.split("_")[2:]])


def parse_epochs(files):
    """Returns {epoch: metrics} for completed epochs (those with an 'Epoch time' line)."""
    epochs = {}
    for f in files:
        epoch, cur = None, {}
        for raw in f.read_text(errors="ignore").splitlines():
            m = LINE.match(raw)
            if not m:
                continue
            msg = m.group(1)
            if re.fullmatch(r"Epoch \d+", msg):
                epoch, cur = int(msg.split()[1]), {}
            elif epoch is None:
                continue
            elif msg.startswith("Current learning rate:"):
                cur["lr"] = float(msg.split(":")[1])
            elif msg.startswith("train_loss"):
                cur["train_loss"] = float(msg.split()[1])
            elif msg.startswith("val_loss"):
                cur["val_loss"] = float(msg.split()[1])
            elif msg.startswith("Pseudo dice"):
                # "[np.float32(0.71)]" or "[0.71, 0.5]" -> one value per foreground class
                dice = [float(x) for x in FLOAT.findall(msg.replace("np.float32", "").replace("float32", ""))]
                for i, d in enumerate(dice, 1):
                    cur[f"pseudo_dice/class_{i}"] = d
                cur["pseudo_dice/mean"] = sum(dice) / len(dice)
            elif msg.startswith("Epoch time:"):
                cur["epoch_time_s"] = float(FLOAT.search(msg.split(":")[1]).group())
                epochs[epoch] = cur  # later files overwrite repeated epochs
    return epochs


def add_ema(epochs):
    """nnU-Net's EMA pseudo Dice (0.9 * prev + 0.1 * current) and its running best."""
    ema, best = None, None
    for e in sorted(epochs):
        d = epochs[e]["pseudo_dice/mean"]
        ema = d if ema is None else 0.9 * ema + 0.1 * d
        best = ema if best is None else max(best, ema)
        epochs[e]["ema_pseudo_dice"] = ema
        epochs[e]["best_ema_pseudo_dice"] = best
        epochs[e]["new_best_ema"] = int(ema == best)


def sync_fold(fold, fold_dir, args, synced):
    """Logs new epochs of one fold. Returns True if the fold is finished and fully synced."""
    files = log_files(fold_dir)
    if not files:
        return False
    epochs = parse_epochs(files)
    add_ema(epochs)
    summary_json = fold_dir / "validation" / "summary.json"
    finished = (fold_dir / "checkpoint_final.pth").exists() and summary_json.exists()

    last = synced.get(fold)
    if last is not None and (not epochs or max(epochs) <= last) and not finished:
        return False

    run = wandb.init(project=args.project, entity=args.entity, id=f"{args.run_prefix}_fold{fold}",
                     name=f"{args.run_prefix}_fold{fold}", group=args.run_prefix, job_type="nnunet_train",
                     resume="allow", dir=str(args.wandb_dir),
                     config={"fold": fold, "results_dir": str(fold_dir)})
    last = int(run.summary.get("epoch", -1)) if last is None else last
    new = [e for e in sorted(epochs) if e > last]
    for e in new:
        run.log({**epochs[e], "epoch": e}, step=e)
    if new:
        last = new[-1]
    run.summary["job_segments"] = len(files)

    done = False
    if finished:
        metrics = json.loads(summary_json.read_text())
        run.summary["final_val/dice_fg_mean"] = metrics["foreground_mean"]["Dice"]
        for cls, m in metrics["mean"].items():
            run.summary[f"final_val/dice_class_{cls}"] = m["Dice"]
        done = True
    run.finish(quiet=True)
    synced[fold] = last
    print(f"{time.strftime('%H:%M:%S')} fold {fold}: logged {len(new)} epochs (up to {last})"
          f"{', final validation logged' if done else ''}", flush=True)
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--results", type=Path, default=RESULTS)
    p.add_argument("--project", default="psmareg_nnunet")
    p.add_argument("--entity", default=None)
    p.add_argument("--run_prefix", default="nnunet_autopet3_lesions")
    p.add_argument("--interval", type=int, default=600, help="seconds between polls")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    args.wandb_dir = args.results / "wandb_sync"
    args.wandb_dir.mkdir(exist_ok=True)

    synced, done = {}, set()
    while True:
        for fold in args.folds:
            if fold in done:
                continue
            try:
                if sync_fold(fold, args.results / f"fold_{fold}", args, synced):
                    done.add(fold)
            except Exception as e:  # keep polling the other folds, e.g. on network hiccups
                print(f"fold {fold}: sync failed: {e!r}", flush=True)
        if args.once or len(done) == len(args.folds):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
