#!/bin/bash
set -e  # Stop script on error

echo "1 0 0 -89.5
0 1 0 -126
0 0 1 -72
0 0 0 1" > affine.txt

filename="${1%%.*}"

mrconvert $1 "$filename"_las.nii.gz -stride -1,2,3 -force
mrtransform -linear affine.txt -flip 0 -replace "$filename"_las.nii.gz "$filename"_ras.nii.gz -force
python ~/research/src/scilpy/scripts/scil_crop_volume.py "$filename"_ras.nii.gz "$filename"_ras_crop.nii.gz --input_bbox $2 --ignore_voxel_size -f

# rm -f "$filename"_las.nii.gz "$filename"_ras.nii.gz affine.txt
