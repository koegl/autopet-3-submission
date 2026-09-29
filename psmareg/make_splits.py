"""Write a patient-grouped 5-fold splits_final.json over all non-test PSMAReg patients.

All sessions of a patient land in the same fold. Test patients (split_journal.json) are never in the dataset.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

SPLIT = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/split_journal.json")
N_FOLDS = 5
SEED = 12345  # nnU-Net's default split seed

dataset = maybe_convert_to_dataset_name(int(sys.argv[1]))
test_patients = set(json.loads(SPLIT.read_text())["test"])
cases = sorted(p.name[:-len(".nii.gz")] for p in (Path(nnUNet_raw) / dataset / "labelsTr").glob("*.nii.gz"))
assert not any(c.split("_")[1] in test_patients for c in cases), "test patient found in dataset"

patients = np.array(sorted({c.split("_")[1] for c in cases}))
splits = []
for fold, (tr, va) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(patients)):
    val_patients = set(patients[va])
    val = [c for c in cases if c.split("_")[1] in val_patients]
    train = [c for c in cases if c.split("_")[1] not in val_patients]
    splits.append({"train": train, "val": val})
    print(f"fold {fold}: {len(train)} train / {len(val)} val cases, {len(tr)} / {len(va)} patients")
(Path(nnUNet_preprocessed) / dataset / "splits_final.json").write_text(json.dumps(splits, indent=2))
