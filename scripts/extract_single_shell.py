import os
import argparse
import numpy as np
import nibabel as nib

from os.path import join as pjoin

from dipy.core.gradients import gradient_table_from_bvals_bvecs
from dipy.io.gradients import read_bvals_bvecs


def build_argparser():
    DESCRIPTION = "Extract 1 one shell from DWI data."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('dwi', type=str, help="diffusion weighted images (.nii|.nii.gz).")
    p.add_argument('b_value', type=int, help="b-value to extract.")
    p.add_argument('--bvals', type=str, help="file containing the b-value associated to each diffusion direction.")
    p.add_argument('--bvecs', type=str, help="file containing the vector associated to each diffusion direction.")
    p.add_argument('--out', type=str, default='.', help="output folder. Default:='%(default)s'.")
    p.add_argument('--basename', type=str, default='dwi', help="basename that will be used to generate ouputs files. Default: %(default)s")
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    dwi_name = args.dwi.split('.')[0]
    bvals_filename = dwi_name + ".bvals"
    bvecs_filename = dwi_name + ".bvecs"
    if args.bvals is not None:
        bvals_filename = args.bvals

    if args.bvecs is not None:
        bvecs_filename = args.bvecs

    bvals, bvecs = read_bvals_bvecs(bvals_filename, bvecs_filename)
    gtab = gradient_table_from_bvals_bvecs(bvals, bvecs)
    idx = gtab.bvals == 1000
    idx[np.logical_and(1000-15 <= gtab.bvals, gtab.bvals <= 1000+15)] = True

    idx[0] = True
    b0_idx = gtab.bvals <= 15
    print("{} b=0 images will be averaged.".format(np.sum(b0_idx)))

    # Make sure we have the right numbers of each bval.
    #assert np.sum(idx) == 91
    #assert np.sum(np.logical_and(2000-15 <= gtab.bvals, gtab.bvals <= 2000+15)) == 90
    #assert np.sum(np.logical_and(3000-20 <= gtab.bvals, gtab.bvals <= 3000+20)) == 90

    new_bvals = gtab.bvals[idx]
    assert new_bvals[0] <= 5
    new_bvals[0] = 0  # Make sure it is 0
    new_bvecs = gtab.bvecs[idx]

    dwi = nib.load(args.dwi)
    b0 = np.mean(dwi.get_data()[:, :, :, b0_idx], axis=-1)
    nii = nib.Nifti1Image(b0, affine=dwi.affine, header=dwi.header)
    out_dir = args.out
    try:
        os.makedirs(out_dir)
    except:
        pass  # Assume it already exists.

    # Write b0
    nib.save(nii, pjoin(out_dir, args.basename + "_b0.nii.gz"))

    new_dwi = dwi.get_data()[:, :, :, idx]
    new_dwi[:, :, :, 0] = b0.copy()
    nii = nib.Nifti1Image(new_dwi, affine=dwi.affine, header=dwi.header)
    dwi_out_name = args.basename + "_b{}".format(args.b_value)
    nib.save(nii, pjoin(out_dir, dwi_out_name + ".nii.gz"))
    open(pjoin(out_dir, dwi_out_name + ".bvecs"), 'w').write("\n".join(map(lambda bv: " ".join(map(str, bv)), new_bvecs.T)))
    open(pjoin(out_dir, dwi_out_name + ".bvals"), 'w').write(" ".join(map(str, new_bvals)))

if __name__ == "__main__":
    main()
