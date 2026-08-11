"""Plan, preprocess and train an nnU-Net on the PSMA PET lesion dataset.

Run nnunet_prepare_dataset.py first.

    python nnunet_train.py                # plan + preprocess + train fold 0
    python nnunet_train.py --skip-preprocess
    python nnunet_train.py --trainer nnUNetTrainer    # constant patch size baseline

nnUNet_preprocessed and nnUNet_results live on the data drive: preprocessing
writes a float32 copy of every case (~85 MB each, ~58 GB unpacked for 560
cases), which does not belong in the repo.

Evaluate with checkpoint_final, not checkpoint_best: the validation split is the
held-out set the three segmentation methods are compared on, so selecting a
checkpoint on it would tilt the comparison towards nnU-Net.
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = "/home/iml/fryderyk.koegl/code/autopet-3-submission"
NNUNET_RAW = os.path.join(REPO_ROOT, "nnUNet_raw")
NNUNET_BASE = "/home/iml/fryderyk.koegl/data/PSMAReg-nnunet"
NNUNET_PREPROCESSED = os.path.join(NNUNET_BASE, "nnUNet_preprocessed")
NNUNET_RESULTS = os.path.join(NNUNET_BASE, "nnUNet_results")

DATASET_ID = "501"
DATASET_NAME = "Dataset501_PSMALesion"
CONFIGURATION = "3d_fullres"
FOLD = "0"
DEFAULT_TRAINER = "nnUNetTrainer_PGPSplus"

# Stop after preprocessing by default: preprocessing needs no GPU and can run
# alongside another training job, whereas training cannot. Flip this to False
# (or pass --no-preprocess-only) once the GPU is free.
PREPROCESS_ONLY = True


def nnunet_env() -> dict:
    env = os.environ.copy()
    env["nnUNet_raw"] = NNUNET_RAW
    env["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    env["nnUNet_results"] = NNUNET_RESULTS
    return env


def run(cmd: list, env: dict) -> None:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--fold", default=FOLD)
    parser.add_argument("--configuration", default=CONFIGURATION)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument(
        "--preprocess-only",
        action=argparse.BooleanOptionalAction,
        default=PREPROCESS_ONLY,
        help="plan and preprocess, then stop without training "
        f"(default: {PREPROCESS_ONLY})",
    )
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="resume from checkpoint_latest.pth",
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        help="also save softmax outputs for the validation set",
    )
    parser.add_argument(
        "--gpu-memory-target",
        type=float,
        default=None,
        help="VRAM budget in GB the planner sizes the patch size "
        "against (nnU-Net default: 8). Lower it to leave room "
        "for another job on the same GPU. Writes a separately "
        "named plans file so the default plans are untouched.",
    )
    args = parser.parse_args()

    # nnU-Net warns that changing the memory target without renaming the plans
    # silently overwrites nnUNetPlans, so derive a distinct name and train from it.
    plans_name = "nnUNetPlans"
    if args.gpu_memory_target is not None:
        plans_name = f"nnUNetPlans_{args.gpu_memory_target:g}G"

    for path in (NNUNET_PREPROCESSED, NNUNET_RESULTS):
        os.makedirs(path, exist_ok=True)

    dataset_dir = os.path.join(NNUNET_RAW, DATASET_NAME)
    if not os.path.isdir(dataset_dir):
        sys.exit(f"{dataset_dir} not found -- run nnunet_prepare_dataset.py first")

    env = nnunet_env()

    splits = os.path.join(NNUNET_PREPROCESSED, DATASET_NAME, "splits_final.json")
    if not os.path.isfile(splits):
        sys.exit(f"{splits} not found -- run nnunet_prepare_dataset.py first")

    if not args.skip_preprocess:
        # -c 3d_fullres: we only train 3d_fullres, so don't preprocess 2d/lowres.
        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            DATASET_ID,
            "-c",
            args.configuration,
            "--verify_dataset_integrity",
        ]
        if args.gpu_memory_target is not None:
            cmd += [
                "-gpu_memory_target",
                str(args.gpu_memory_target),
                "-overwrite_plans_name",
                plans_name,
            ]
        run(cmd, env)

    if args.preprocess_only:
        print(f"\npreprocessed: {os.path.join(NNUNET_PREPROCESSED, DATASET_NAME)}")
        print("stopping before training (--no-preprocess-only to train)")
        return

    train_cmd = [
        "nnUNetv2_train",
        DATASET_ID,
        args.configuration,
        args.fold,
        "-tr",
        args.trainer,
        "-p",
        plans_name,
    ]
    if args.continue_training:
        train_cmd.append("--c")
    if args.npz:
        train_cmd.append("--npz")
    run(train_cmd, env)

    print(f"\nresults: {os.path.join(NNUNET_RESULTS, DATASET_NAME)}")


if __name__ == "__main__":
    main()
