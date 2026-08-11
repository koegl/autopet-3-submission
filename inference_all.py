import csv
import os
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import tqdm

from main import build_predictor, run_inference_in_memory

IMAGES_DIR = "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr"
LABELS_DIR = "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr"
OUTPUT_DIR = "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr_autopet_fold0"
CSV_PATH = os.path.join(OUTPUT_DIR, "dice_autopet_fold0.csv")
MODEL_FOLDER = (
    "/home/iml/fryderyk.koegl/code/autopet-3-submission/"
    "autoPET-3-LesionTracer/Dataset222_AutoPETIII_2024/"
    "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"
)
FOLDS = (0,)
USE_MIRRORING = False  # 3D mirror TTA runs every tile 8x; off is ~8x faster
CSV_FIELDS = ["filename", "dice", "gt_voxels", "pred_voxels"]
MEAN_ROW_NAME = "MEAN"
MEAN_NONEMPTY_ROW_NAME = "MEAN_NONEMPTY"
SUMMARY_ROW_NAMES = (MEAN_ROW_NAME, MEAN_NONEMPTY_ROW_NAME)


def collect_cases() -> List[Tuple[str, str, str, str]]:
    """Return (name, ct_path, pet_path, label_path) for every PET label."""
    cases = []
    for name in sorted(os.listdir(LABELS_DIR)):
        if "_0001_" not in name or not name.endswith(".nii.gz"):
            continue
        pet_path = os.path.join(IMAGES_DIR, name)
        ct_path = os.path.join(IMAGES_DIR, name.replace("_0001_", "_0000_"))
        if not (os.path.isfile(pet_path) and os.path.isfile(ct_path)):
            continue
        cases.append((name, ct_path, pet_path, os.path.join(LABELS_DIR, name)))
    return cases


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt > 0
    pred = pred > 0
    total = gt.sum() + pred.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(gt, pred).sum() / total)


def read_done(csv_path: str) -> set:
    """Filenames already scored (the trailing summary rows are ignored)."""
    if not os.path.isfile(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {
            row["filename"]
            for row in csv.DictReader(f)
            if row["filename"] not in SUMMARY_ROW_NAMES
        }


def append_row(csv_path: str, row: dict) -> None:
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def is_empty_case(gt_voxels: int, pred_voxels: int) -> bool:
    """True when GT and prediction are both empty, i.e. Dice is trivially 1.0."""
    return gt_voxels == 0 and pred_voxels == 0


def strip_summary_rows(csv_path: str) -> List[dict]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["filename"] not in SUMMARY_ROW_NAMES]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary_rows(csv_path: str) -> Tuple[Optional[float], Optional[float]]:
    rows = strip_summary_rows(csv_path)
    if not rows:
        return None, None

    all_dices = [float(r["dice"]) for r in rows]
    nonempty_dices = [
        float(r["dice"])
        for r in rows
        if not is_empty_case(int(r["gt_voxels"]), int(r["pred_voxels"]))
    ]

    mean_dice = float(np.mean(all_dices))
    append_row(
        csv_path,
        {
            "filename": MEAN_ROW_NAME,
            "dice": f"{mean_dice:.6f}",
            "gt_voxels": len(all_dices),
            "pred_voxels": "",
        },
    )

    mean_nonempty = float(np.mean(nonempty_dices)) if nonempty_dices else None
    append_row(
        csv_path,
        {
            "filename": MEAN_NONEMPTY_ROW_NAME,
            "dice": "" if mean_nonempty is None else f"{mean_nonempty:.6f}",
            "gt_voxels": len(nonempty_dices),
            "pred_voxels": "",
        },
    )
    return mean_dice, mean_nonempty


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cases = collect_cases()
    done = read_done(CSV_PATH)
    strip_summary_rows(CSV_PATH)

    predictor = build_predictor(
        MODEL_FOLDER, FOLDS, torch.device("cuda"), use_mirroring=USE_MIRRORING
    )

    dices: List[float] = []
    dices_nonempty: List[float] = []
    pbar = tqdm.tqdm(cases, ncols=150, desc="autoPET fold0")
    for name, ct_path, pet_path, label_path in pbar:
        out_path = os.path.join(OUTPUT_DIR, name)

        if os.path.isfile(out_path) and name in done:
            pbar.set_postfix_str(f"{name} skipped")
            continue

        pet_img = nib.load(pet_path)
        if os.path.isfile(out_path):
            # Segmentation exists but was never scored -- reuse it.
            seg = nib.load(out_path).get_fdata().astype(np.uint8)
        else:
            seg = run_inference_in_memory(predictor, ct_path, pet_path).astype(np.uint8)
            nib.save(
                nib.Nifti1Image(seg, pet_img.affine, pet_img.header),
                out_path,
            )

        gt = nib.load(label_path).get_fdata()
        dice = dice_score(gt, seg)
        gt_voxels = int((gt > 0).sum())
        pred_voxels = int((seg > 0).sum())

        dices.append(dice)
        if not is_empty_case(gt_voxels, pred_voxels):
            dices_nonempty.append(dice)

        append_row(
            CSV_PATH,
            {
                "filename": name,
                "dice": f"{dice:.6f}",
                "gt_voxels": gt_voxels,
                "pred_voxels": pred_voxels,
            },
        )
        mean_nonempty = f"{np.mean(dices_nonempty):.3f}" if dices_nonempty else "n/a"
        pbar.set_postfix_str(
            f"dice={dice:.3f} mean={np.mean(dices):.3f} "
            f"mean_ne={mean_nonempty} (n={len(dices_nonempty)}/{len(dices)})"
        )
    pbar.close()

    mean_dice, mean_nonempty = write_summary_rows(CSV_PATH)
    if mean_dice is None:
        print("No cases scored.")
    else:
        print(f"Mean DSC over all images:        {mean_dice:.4f}")
        if mean_nonempty is None:
            print("Mean DSC excluding empty cases: n/a (every case was empty)")
        else:
            print(f"Mean DSC excluding empty cases: {mean_nonempty:.4f}")
        print(f"-> {CSV_PATH}")


if __name__ == "__main__":
    main()
