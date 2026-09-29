"""Build the nnU-Net raw datasets (lesions + organs) for autoPET3-style training on PSMAReg.

Cases: every session of every patient not in the test split (val patients are included, the
train/val separation happens in splits_final.json). Case id = PSMARegPSMA_<patient>_<session>.

Organ labels follow the autoPET3 scheme (10 classes): TotalSegmentator `total` labels from the
PSMAReg labelsTr plus salivary glands from TotalSegmentator `head_glands_cavities`. Rerun this script
once all gland predictions exist; cases without them are reported and written without glands.
"""
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import SimpleITK as sitk

SRC = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset")
GLANDS = SRC / "labelsTr_head_glands_cavities"
SPLIT = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/split_journal.json")
RAW = Path("/home/iml/fryderyk.koegl/data/PSMAReg/lesion_seg_nnunet_autopet3/nnUNet_raw")
LESIONS = RAW / "Dataset901_PSMAReg_lesions"
ORGANS = RAW / "Dataset902_PSMAReg_organs"

# TotalSegmentator `total` (v2) -> autoPET3 organ classes
TOTAL_TO_ORGAN = {
    1: 1,  # spleen
    2: 2, 3: 2,  # kidneys
    5: 3,  # liver
    21: 4,  # urinary bladder
    10: 5, 11: 5, 12: 5, 13: 5, 14: 5,  # lung lobes
    90: 6,  # brain
    51: 7,  # heart
    6: 8,  # stomach
    22: 9,  # prostate
}
# TotalSegmentator `head_glands_cavities`: parotid L/R, submandibular R/L -> glands
GLANDS_TO_ORGAN = {7: 10, 8: 10, 9: 10, 10: 10}
ORGAN_NAMES = ["background", "spleen", "kidneys", "liver", "urinary_bladder", "lungs", "brain", "heart",
               "stomach", "prostate", "salivary_glands"]


def remap(arr, mapping):
    lut = np.zeros(max(int(arr.max()), max(mapping)) + 1, dtype=np.uint8)
    for src, dst in mapping.items():
        lut[src] = dst
    return lut[arr]


def write_label(arr, ref, path):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(path), useCompression=True)


def link_images(ds, patient, session, case):
    for channel in ("0000", "0001"):
        dst = ds / "imagesTr" / f"{case}_{channel}.nii.gz"
        if not dst.is_symlink():
            dst.symlink_to(SRC / "imagesTr" / f"PSMARegPSMA_{patient}_{channel}_{session}.nii.gz")


def process(key):
    """Returns True if the salivary gland prediction was available."""
    patient, session = key
    case = f"PSMARegPSMA_{patient}_{session}"
    for ds in (LESIONS, ORGANS):
        link_images(ds, patient, session, case)

    lesion = sitk.ReadImage(str(SRC / "labelsTr" / f"PSMARegPSMA_{patient}_0001_{session}.nii.gz"))
    write_label(sitk.GetArrayFromImage(lesion) > 0, lesion, LESIONS / "labelsTr" / f"{case}.nii.gz")

    total = sitk.ReadImage(str(SRC / "labelsTr" / f"PSMARegPSMA_{patient}_0000_{session}.nii.gz"))
    organs = remap(sitk.GetArrayFromImage(total), TOTAL_TO_ORGAN)
    glands_path = GLANDS / f"PSMARegPSMA_{patient}_0000_{session}.nii.gz"
    has_glands = glands_path.exists()
    if has_glands:
        glands = remap(sitk.GetArrayFromImage(sitk.ReadImage(str(glands_path))), GLANDS_TO_ORGAN)
        organs[glands > 0] = glands[glands > 0]
    write_label(organs, total, ORGANS / "labelsTr" / f"{case}.nii.gz")
    return has_glands


def dataset_json(labels, n):
    # both channels as "CT" -> CTNormalization, as in the autoPET3 plans
    return {"channel_names": {"0": "CT", "1": "CT"}, "labels": labels, "numTraining": n,
            "file_ending": ".nii.gz", "description": "PSMAReg sessions (CT, PSMA PET), test patients excluded"}


if __name__ == "__main__":
    test = set(json.loads(SPLIT.read_text())["test"])
    keys = sorted({(p.name.split("_")[1], p.name.split("_")[3][:2])
                   for p in (SRC / "imagesTr").glob("PSMARegPSMA_0*_0000_*.nii.gz")})
    keys = [k for k in keys if k[0] not in test]
    print(f"{len(keys)} cases from {len({k[0] for k in keys})} patients")

    for ds in (LESIONS, ORGANS):
        (ds / "imagesTr").mkdir(parents=True, exist_ok=True)
        (ds / "labelsTr").mkdir(parents=True, exist_ok=True)
    with Pool(16) as pool:
        has_glands = pool.map(process, keys)

    (LESIONS / "dataset.json").write_text(json.dumps(dataset_json({"background": 0, "lesion": 1}, len(keys)), indent=2))
    (ORGANS / "dataset.json").write_text(
        json.dumps(dataset_json({n: i for i, n in enumerate(ORGAN_NAMES)}, len(keys)), indent=2))
    print(f"salivary glands missing for {has_glands.count(False)}/{len(keys)} cases")
