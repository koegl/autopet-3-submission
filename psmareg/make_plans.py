"""Adapt the autoPET3 MultiTalent ResEncL plans (192^3 patch, pretrained architecture) to a PSMAReg dataset.

Architecture, patch size and batch size stay as in autoPET3 so the pretrained weights load; spacing, image
size and intensity statistics come from our dataset's default nnUNetPlans.
"""
import json
import sys
from pathlib import Path

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

dataset = maybe_convert_to_dataset_name(int(sys.argv[1]))
template = Path(nnunetv2.__file__).parent / "architecture" / "nnUNetResEncUNetLPlansMultiTalent.json"
plans = json.loads(template.read_text())
ours = json.loads((Path(nnUNet_preprocessed) / dataset / "nnUNetPlans.json").read_text())

plans["dataset_name"] = dataset
for key in ("original_median_spacing_after_transp", "original_median_shape_after_transp",
            "transpose_forward", "transpose_backward", "foreground_intensity_properties_per_channel"):
    plans[key] = ours[key]
for key in ("spacing", "median_image_size_in_voxels"):
    plans["configurations"]["3d_fullres"][key] = ours["configurations"]["3d_fullres"][key]

out = Path(nnUNet_preprocessed) / dataset / f"{plans['plans_name']}.json"
out.write_text(json.dumps(plans, indent=4))
print(f"wrote {out}")
