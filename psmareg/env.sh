# source psmareg/env.sh  (works on the workstation and on the HPC cluster)
if [ -d /lustre/groups/iml/data/PSMAReg ]; then
    R=/lustre/groups/iml/data/PSMAReg/lesion_seg_nnunet_autopet3
else
    R=/home/iml/fryderyk.koegl/data/PSMAReg/lesion_seg_nnunet_autopet3
fi
export nnUNet_raw=$R/nnUNet_raw
export nnUNet_preprocessed=$R/nnUNet_preprocessed
export nnUNet_results=$R/nnUNet_results
export PRETRAINED=$R/pretrained/Dataset619_nativemultistem/MultiTalent_trainer_multistems_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/checkpoint_final.pth
source /home/iml/fryderyk.koegl/code/autopet-3-submission/.venv/bin/activate
