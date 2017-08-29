#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import argparse
import numpy as np

import nibabel as nib

from learn2track.utils import Timer
from learn2track.neurotools import TractographyData


def buildArgsParser():
    DESCRIPTION = """Script to split a dataset while making sure the split is similar every bundle.

        Examples
        --------
        split_dataset.py ismrm15_challenge.npz -v
        split_dataset.py ismrm15_challenge.npz -v --leave-one-out 0 1 2 3,4 5,6 7,8 9 10,11 12,13 14 15,16 17,18 19,20 21,22 23,24
    """
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('dataset', help='training data (.npz).')
    p.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   help='respectively the sizes of the split for trainset, validset and testset. Default: %(default)s')
    p.add_argument('--split_type', choices=["percentage", "count"], default="percentage",
                   help='type of the split, either use percents or fixed counts. Default: %(default)s')
    p.add_argument('--seed', type=int, default=1234, help='seed to use to shuffle data. Default: %(default)s')
    p.add_argument('--delete', action="store_true", help='if specified, delete bundle file after being splitted.')
    p.add_argument('--leave-one-out', metavar="BUNDLE_ID", nargs="+",
                   help='list of bundle_id (use comma to merge multiple into one), from which a leave-one-out split will be generated (i.e. remove one bundle from the trainset and generate a testset and a validset from it.')

    p.add_argument('--list-bundles-name', action="store_true",
                   help='if specified, list the bundles name and quit.')

    p.add_argument('-v', '--verbose', action="store_true", help='verbose mode')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    data = TractographyData.load(args.dataset)
    streamlines = data.streamlines
    print("{} has {:,} streamlines".format(args.dataset, len(streamlines)))

    if args.list_bundles_name:
        for bundle_name in data.bundle_names:
            bundle_id = data.name2id[bundle_name]
            print("{}: {}".format(bundle_id, bundle_name))

        return

    if args.leave_one_out is not None:
        with Timer("Splitting {} using a leave-one-out strategy".format(args.dataset), newline=True):
            for bundle in args.leave_one_out:
                rng = np.random.RandomState(args.seed)
                train_data = TractographyData(data.signal, data.gradients, data.name2id)
                valid_data = TractographyData(data.signal, data.gradients, data.name2id)
                test_data = TractographyData(data.signal, data.gradients, data.name2id)

                bundle_ids_to_exclude = list(map(int, bundle.split(',')))
                missing_bundles_name = [data.bundle_names[i] for i in bundle_ids_to_exclude]

                if args.verbose:
                    print("Leaving out {}...".format(", ".join(missing_bundles_name)))

                include = np.ones(len(data.bundle_ids), dtype=bool)
                exclude = np.zeros(len(data.bundle_ids), dtype=bool)
                for i in bundle_ids_to_exclude:
                    include = np.logical_and(include, data.bundle_ids != i)
                    exclude = np.logical_or(exclude, data.bundle_ids == i)

                include_idx = np.where(include)[0]
                exclude_idx = np.where(exclude)[0]
                rng.shuffle(include_idx)
                rng.shuffle(exclude_idx)

                trainset_indices = include_idx
                validset_indices = exclude_idx[:len(exclude_idx)//2]
                testset_indices = exclude_idx[len(exclude_idx)//2:]

                train_data.add(streamlines[trainset_indices], bundle_ids=data.bundle_ids[trainset_indices])
                valid_data.add(streamlines[validset_indices], bundle_ids=data.bundle_ids[validset_indices])
                test_data.add(streamlines[testset_indices], bundle_ids=data.bundle_ids[testset_indices])

                filename = "missing_{}.npz".format("_".join(missing_bundles_name))
                with Timer("Saving dataset: {}".format(filename[:-4])):
                    train_data.save(filename[:-4] + "_trainset.npz")
                    valid_data.save(filename[:-4] + "_validset.npz")
                    test_data.save(filename[:-4] + "_testset.npz")

    else:
        rng = np.random.RandomState(args.seed)
        train_data = TractographyData(data.signal, data.gradients, data.name2id)
        valid_data = TractographyData(data.signal, data.gradients, data.name2id)
        test_data = TractographyData(data.signal, data.gradients, data.name2id)

        with Timer("Splitting {} as follow {} using {}".format(args.dataset, args.split, args.split_type), newline=args.verbose):
            for bundle_name in data.bundle_names:
                if args.verbose:
                    print("Splitting bundle {}...".format(bundle_name))

                bundle_id = data.name2id[bundle_name]
                indices = np.where(data.bundle_ids == bundle_id)[0]
                nb_examples = len(indices)
                rng.shuffle(indices)

                if args.split_type == "percentage":
                    trainset_size = int(np.round(args.split[0] * nb_examples))
                    validset_size = int(np.round(args.split[1] * nb_examples))
                    testset_size = int(np.round(args.split[2] * nb_examples))
                    # Make sure the splits sum to nb_examples
                    testset_size += nb_examples - (trainset_size + validset_size + testset_size)
                elif args.split_type == "count":
                    raise NotImplementedError("Split type `count` not implemented yet!")

                assert trainset_size + validset_size + testset_size == nb_examples

                trainset_indices = indices[:trainset_size]

                if testset_size > 0:
                    validset_indices = indices[trainset_size:-testset_size]
                    testset_indices = indices[-testset_size:]
                else:
                    validset_indices = indices[trainset_size:]
                    testset_indices = []

                train_data.add(streamlines[trainset_indices], bundle_name)
                valid_data.add(streamlines[validset_indices], bundle_name)
                test_data.add(streamlines[testset_indices], bundle_name)

        with Timer("Saving"):
            train_data.save(args.dataset[:-4] + "_trainset.npz")
            valid_data.save(args.dataset[:-4] + "_validset.npz")
            test_data.save(args.dataset[:-4] + "_testset.npz")

        if args.delete:
            os.remove(args.dataset)


if __name__ == '__main__':
    main()
