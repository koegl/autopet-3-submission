"""Progressive Growing of Patch Size (PGPS+) curriculum trainer.

Ported from Stefan Fischer's MICCAI 2024 implementation
(https://github.com/compai-lab/2024-miccai-fischer, paper: arXiv:2407.07853)
to nnU-Net v2.5.1. The upstream code targets nnU-Net 2.0.0 only.

Differences from upstream, all forced by the 2.5.1 API or by resource safety:

* `ConfigurationManager.num_pool_per_axis` no longer exists (the architecture
  spec moved into `architecture.arch_kwargs`); it is now derived from the
  strides. `set_patch_size` / `set_batch_size` were added as well. See
  nnunetv2/utilities/plans_handling/plans_handler.py.
* The stage schedule is precomputed once and indexed by epoch instead of being
  advanced incrementally. This is the same sequence of patch sizes, but it
  survives resuming from a checkpoint -- upstream restarts the curriculum from
  the smallest patch size whenever training is resumed.
* Dataloaders from `on_train_start`, and the previous stage's dataloader, are
  shut down rather than leaked. Upstream leaves a set of worker processes alive
  per stage.

Method: start at the smallest patch size the architecture allows, grow it in
minimal steps over training, and always validate at the final (planned) patch
size, which is also the inference patch size.
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_PGPSplus(nnUNetTrainer):
    # Foreground oversampling is raised during the curriculum: small patches on
    # a dataset with sparse lesions would otherwise almost always be background.
    curriculum_oversample_foreground_percent = 0.5

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_iterations_per_epoch = 250

    # ------------------------------------------------------------------ schedule

    def _plan_patch_sizes(self) -> List[np.ndarray]:
        """Patch sizes for every curriculum stage, smallest to planned.

        Grows one axis at a time, cycling through axes, in steps of
        2**num_pool_per_axis so every stage stays divisible by the network's
        total downsampling factor.
        """
        step = np.array([2 ** p for p in self.num_pool_per_axis])
        patch_sizes = [self.min_patch_size.copy()]
        current = self.min_patch_size.copy()
        i = 1
        while not (current == self.original_patch_size).all():
            grown = current.copy()
            while (grown == current).all():
                if (grown == self.original_patch_size).all():
                    break
                add = np.zeros(3, dtype=int)
                add[i % 3] = 1
                grown = np.minimum(grown + add * step, self.original_patch_size)
                i += 1
            current = grown
            patch_sizes.append(current.copy())
        return patch_sizes

    def _plan_batch_sizes(self, patch_sizes: List[np.ndarray]) -> List[int]:
        """Largest batch size per stage such that voxels per batch never decrease.

        The final stage keeps the planned batch size (which nnU-Net chose to fit
        the planned patch size in GPU memory), and earlier, smaller patches get
        proportionally larger batches.
        """
        batch_sizes = [self.original_batch_size]
        for i in range(len(patch_sizes) - 2, -1, -1):
            ratio = np.prod(patch_sizes[i + 1]) / np.prod(patch_sizes[i])
            batch_sizes.append(int(ratio * batch_sizes[-1]))
        batch_sizes = [max(b, self.original_batch_size) for b in batch_sizes]
        return batch_sizes[::-1]

    def _stage_for_epoch(self, epoch: int) -> int:
        return int(min(epoch // self.epochs_per_stage, self.num_stages - 1))

    # ------------------------------------------------------------------ plumbing

    @staticmethod
    def _shutdown(dataloader) -> None:
        if isinstance(dataloader, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
            old_stdout = sys.stdout
            with open(os.devnull, 'w') as f:
                sys.stdout = f
                dataloader._finish()
                sys.stdout = old_stdout

    def _build_train_loader_for_stage(self, stage: int) -> None:
        patch_size = self.patch_sizes[stage]
        batch_size = self.batch_sizes[stage]

        self.configuration_manager.set_patch_size(patch_size)
        self.configuration_manager.set_batch_size(batch_size)
        self.batch_size = batch_size

        previous = getattr(self, 'dataloader_train', None)
        self.dataloader_train, _ = self.get_dataloaders()
        if previous is not None:
            self._shutdown(previous)

        self.patch_size = patch_size
        self.print_to_log_file(
            f'Curriculum stage {stage + 1}/{self.num_stages}: '
            f'patch size {list(patch_size)}, batch size {batch_size}'
        )

    def _build_val_loader(self) -> None:
        """Validation always runs at the planned (inference) patch size."""
        self.configuration_manager.set_patch_size(self.original_patch_size)
        self.configuration_manager.set_batch_size(self.original_batch_size)
        self.batch_size = self.original_batch_size
        _, self.dataloader_val = self.get_dataloaders()

    # ------------------------------------------------------------------ training

    def run_training(self):
        self.on_train_start()

        # on_train_start built dataloaders at the planned patch size; the
        # curriculum replaces both, so release those workers now.
        self._shutdown(self.dataloader_train)
        self._shutdown(self.dataloader_val)
        self.dataloader_train = None
        self.dataloader_val = None

        self.original_patch_size = np.array(self.configuration_manager.patch_size)
        self.original_batch_size = self.configuration_manager.batch_size
        self.num_pool_per_axis = self.configuration_manager.num_pool_per_axis
        # Smallest patch that still leaves a non-degenerate bottleneck.
        self.min_patch_size = np.array([
            2 ** (self.num_pool_per_axis[0] + 1),
            2 ** self.num_pool_per_axis[1],
            2 ** self.num_pool_per_axis[2],
        ])
        self.min_patch_size = np.minimum(self.min_patch_size, self.original_patch_size)

        self.patch_sizes = self._plan_patch_sizes()
        self.batch_sizes = self._plan_batch_sizes(self.patch_sizes)
        self.num_stages = len(self.patch_sizes)
        self.epochs_per_stage = max(1, self.num_epochs // self.num_stages)

        self.print_to_log_file('##### Progressive Growing of Patch Size (PGPS+) #####')
        self.print_to_log_file(f'Planned patch size:  {list(self.original_patch_size)}')
        self.print_to_log_file(f'Planned batch size:  {self.original_batch_size}')
        self.print_to_log_file(f'num_pool_per_axis:   {self.num_pool_per_axis}')
        self.print_to_log_file(f'Minimal patch size:  {list(self.min_patch_size)}')
        self.print_to_log_file(f'Stages:              {self.num_stages} '
                               f'({self.epochs_per_stage} epochs each)')
        for s, (ps, bs) in enumerate(zip(self.patch_sizes, self.batch_sizes)):
            self.print_to_log_file(f'  stage {s:2d}: patch {list(ps)} batch {bs}')
        self.print_to_log_file('####################################################')

        self.oversample_foreground_percent = self.curriculum_oversample_foreground_percent

        self._build_val_loader()
        current_stage = None

        for epoch in range(self.current_epoch, self.num_epochs):
            stage = self._stage_for_epoch(epoch)
            if stage != current_stage:
                if current_stage is not None:
                    # Snapshot the weights at the end of each patch size phase.
                    ps = self.patch_sizes[current_stage]
                    self.save_checkpoint(join(
                        self.output_folder,
                        f'checkpoint_{ps[0]}x{ps[1]}x{ps[2]}.pth'))
                self._build_train_loader_for_stage(stage)
                current_stage = stage

            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        # Restore the planned configuration so the final validation and all
        # downstream inference use the planned (inference) patch size.
        self.configuration_manager.set_patch_size(self.original_patch_size)
        self.configuration_manager.set_batch_size(self.original_batch_size)
        self.batch_size = self.original_batch_size
        self.oversample_foreground_percent = 0.33  # nnU-Net default
        self.on_train_end()
