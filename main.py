import contextlib
import functools
import os
from typing import Tuple

import nibabel as nib
import numpy as np
import torch
import tqdm

from nnunetv2.inference import predict_from_raw_data

torch.load = functools.partial(torch.load, weights_only=False)


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def run_inference_in_memory(
    predictor: predict_from_raw_data.nnUNetPredictor,
    ct_path: str,
    pet_path: str,
) -> np.ndarray:
    ct_img = nib.load(ct_path)
    pet_img = nib.load(pet_path)

    ct = ct_img.get_fdata().astype(np.float32)
    pet = pet_img.get_fdata().astype(np.float32)
    image = np.stack([ct, pet], axis=0)

    spacing = ct_img.header.get_zooms()[:3]
    props = {"spacing": spacing}
    with suppress_output():
        seg = predictor.predict_single_npy_array(image, props, None, None, False)
    return seg


def build_predictor(
    model_folder: str,
    folds: Tuple[int, ...],
    device: torch.device,
    use_mirroring: bool = True,
) -> predict_from_raw_data.nnUNetPredictor:
    predictor = predict_from_raw_data.nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def run_inference(
    predictor: predict_from_raw_data.nnUNetPredictor,
    ct_path: str,
    pet_path: str,
    output_folder: str,
) -> None:
    list_of_lists = [[ct_path, pet_path]]
    progress_bar = tqdm.tqdm(total=1, desc="Running inference")
    predictor.predict_from_files(
        list_of_lists,
        output_folder,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )
    progress_bar.update(1)
    progress_bar.close()


def main() -> None:
    model_folder = (
        "/home/iml/fryderyk.koegl/code/autopet-3-submission/"
        "autoPET-3-LesionTracer/Dataset222_AutoPETIII_2024/"
        "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"
    )
    ct_path = "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr/PSMARegPSMA_0337_0000_01.nii.gz"
    pet_path = "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr/PSMARegPSMA_0337_0001_01.nii.gz"
    output_folder = "/home/iml/fryderyk.koegl/code/autopet-3-submission/output"
    # folds = (0,)
    folds = (0, 1, 2, 3, 4)
    device = torch.device("cuda")

    predictor = build_predictor(model_folder, folds, device)
    run_inference(predictor, ct_path, pet_path, output_folder)


if __name__ == "__main__":
    main()
