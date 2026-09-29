"""Run TotalSegmentator head_glands_cavities on all CTs of non-test PSMAReg patients."""
import json
import subprocess
from pathlib import Path

from tqdm import tqdm

IMAGES = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr")
OUT = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr_head_glands_cavities")
SPLIT = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/split_journal.json")
TOTALSEG = "/home/iml/fryderyk.koegl/code/venv_totalseg/bin/TotalSegmentator"

test = set(json.loads(SPLIT.read_text())["test"])
cts = sorted(p for p in IMAGES.glob("PSMARegPSMA_0*_0000_*.nii.gz") if p.name.split("_")[1] not in test)
LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
print(f"{len(cts)} CTs (test patients excluded: {len(test)})")

todo = [ct for ct in cts if not (OUT / ct.name).exists()]
for ct in tqdm(todo, desc="head_glands_cavities"):
    out = OUT / ct.name
    tmp = OUT / f"tmp_{ct.name}"  # renamed on success, so interrupted cases are redone
    with open(LOGS / ct.name.replace(".nii.gz", ".log"), "w") as log:
        subprocess.run([TOTALSEG, "-i", str(ct), "-o", str(tmp), "-ta", "head_glands_cavities", "--ml"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    tmp.rename(out)
print("done")
