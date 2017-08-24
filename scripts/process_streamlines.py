#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import textwrap

import nibabel as nib
import numpy as np
from dipy.core.gradients import gradient_table

from learn2track.neurotools import TractographyData, subsample_streamlines
from learn2track.utils import Timer


def build_argparser():
    DESCRIPTION = textwrap.dedent(
        """ Script to generate training data from a list of streamlines bundle files.

            This results in a .npz file containing the following keys:\n"
            'coords': ndarray of shape (N, 3)
                Coordinates of each point of every streamlines expressed in voxel space.
                N is the total number points of all streamlines.
            'offsets': ndarray of shape (M,) with dtype int64
                Index of the beginning of each streamline. M is the total number of streamlines.
            'lengths': ndarray of shape (M,) with dtype int16
                Number of points of each streamline. M is the total number of streamlines.
            'bundle_ids': ndarray of shape (M,) with dtype int16
                ID of the bundle each streamline belongs to
            'name2id': dict
                Mapping between bundle names and bundle IDs.
            'signal': :class:`Nifti1Image` object (from nibabel)
                Diffusion signal
            'gradients': :class:`GradientTable` object (from dipy)
                Diffusion gradients information
        """)
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subsampling_parser = argparse.ArgumentParser(add_help=False)
    subsampling_parser.add_argument('--subsample-streamlines', action='store_true',
                                    help="Downsample every bundle using QuickBundles. "
                                         "A clustering threshold of 6 and a removal distance of 2 are used by default, but can be changed. "
                                         "NOTE: Changing the default values will have no effect if this flag is not given")
    subsampling_parser.add_argument('--clustering_threshold', default=6, help="Threshold used to cluster streamlines before computing distance matrix")
    subsampling_parser.add_argument('--removal_distance', default=2, help="Streamlines closer than this distance will be reduced to a single streamline")

    # General options (optional)
    general_parser = argparse.ArgumentParser(add_help=False)
    general_parser.add_argument('--out', metavar='FILE', default="dataset.npz", help='output filename (.npz). Default: dataset.npz')
    general_parser.add_argument('--dtype', type=str, default="float32", help="'float16' or 'float32'. Default: 'float32'")
    general_parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')

    signal_subparsers = p.add_subparsers(title="Signal source", dest="signal_source")
    signal_subparsers.required = True
    raw_signal_parser = signal_subparsers.add_parser("raw_signal", parents=[subsampling_parser, general_parser],
                                                     description="Use raw signal from a Nifti image")
    signal_parser = raw_signal_parser.add_argument_group("Raw signal arguments")
    signal_parser.add_argument('signal', help='Diffusion signal (.nii|.nii.gz).')
    signal_parser.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of streamlines bundle files.')
    signal_parser.add_argument('--bvals', help='File containing diffusion gradient lengths (Default: guess it from `signal`).')
    signal_parser.add_argument('--bvecs', help='File containing diffusion gradient directions (Default: guess it from `signal`).')

    processed_signal_parser = signal_subparsers.add_parser("processed_signal", parents=[subsampling_parser, general_parser],
                                                           description="Extract signal from a TractographyData (.npz) file, and ignore existing streamlines.")
    signal_parser = processed_signal_parser.add_argument_group("Processed signal arguments")
    signal_parser.add_argument('tracto_data', help="TractographyData file containing the processed signal along existing streamlines and other info. (.npz)")
    signal_parser.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of streamlines bundle files.')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    tracto_data = None

    if args.signal_source == "raw_signal":
        signal = nib.load(args.signal)
        signal.get_data()  # Forces loading volume in-memory.
        basename = re.sub('(\.gz|\.nii.gz)$', '', args.signal)
        bvals = basename + '.bvals' if args.bvals is None else args.bvals
        bvecs = basename + '.bvecs' if args.bvecs is None else args.bvecs

        gradients = gradient_table(bvals, bvecs)
        tracto_data = TractographyData(signal, gradients)
    elif args.signal_source == "processed_signal":
        loaded_tracto_data = TractographyData.load(args.tracto_data)
        tracto_data = TractographyData(loaded_tracto_data.signal, loaded_tracto_data.gradients)

    # Compute matrix that brings streamlines back to diffusion voxel space.
    rasmm2vox_affine = np.linalg.inv(tracto_data.signal.affine)

    # Retrieve data.
    with Timer("Retrieving data", newline=args.verbose):
        for filename in sorted(args.bundles):
            if args.verbose:
                print("{}".format(filename))

            # Load streamlines
            tfile = nib.streamlines.load(filename)
            tractogram = tfile.tractogram
            if args.subsample_streamlines:
                original_streamlines = tractogram.streamlines
                output_streamlines = subsample_streamlines(original_streamlines, args.clustering_threshold,
                                                           args.removal_distance)

                print("Total difference: {} / {}".format(len(original_streamlines), len(output_streamlines)))
                new_tractogram = nib.streamlines.Tractogram(output_streamlines,
                                                            affine_to_rasmm=tractogram.affine_to_rasmm)
                tractogram = new_tractogram

            tractogram.apply_affine(rasmm2vox_affine)

            # Add streamlines to the TractogramData
            bundle_name = os.path.splitext(os.path.basename(filename))[0]
            tracto_data.add(tractogram.streamlines, bundle_name)

    if args.verbose:
        diff = tracto_data.streamlines._data - tracto_data.streamlines._data.astype(args.dtype)
        precision_error = np.sum(np.sqrt(np.sum(diff ** 2, axis=1)))
        avg_precision_error = precision_error / len(tracto_data.streamlines._data)
        print("Precision error: {} (avg. {})".format(precision_error, avg_precision_error))

    # Save streamlines coordinates using either float16 or float32.
    tracto_data.streamlines._data = tracto_data.streamlines._data.astype(args.dtype)

    # Save dataset
    tracto_data.save(args.out)


if __name__ == '__main__':
    main()
