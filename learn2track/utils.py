from __future__ import print_function

import os
import sys
import numpy as np
import theano
import theano.tensor as T
import shutil
import hashlib

from collections import OrderedDict
from time import time
from os.path import join as pjoin

import smartlearner.utils as smartutils


class Timer():
    """ Times code within a `with` statement. """
    def __init__(self, txt, newline=False):
        self.txt = txt
        self.newline = newline

    def __enter__(self):
        self.start = time()
        if not self.newline:
            print(self.txt + "... ", end="")
            sys.stdout.flush()
        else:
            print(self.txt + "... ")

    def __exit__(self, type, value, tb):
        if self.newline:
            print(self.txt + " done in ", end="")

        print("{:.2f} sec.".format(time()-self.start))


def generate_uid_from_string(value):
    """ Creates unique identifier from a string. """
    return hashlib.sha256(value.encode()).hexdigest()


def find_closest(sphere, xyz, normed=True):
    """
    Find the index of the vertex in the Sphere closest to the input vector

    Parameters
    ----------
    xyz : ndarray shape (N, 3)
        Input vector(s)
    normed : {True, False}, optional
        Normalized input vector(s).

    Return
    ------
    idx : ndarray shape (N,)
        The index/indices into the Sphere.vertices array that gives the closest
        vertex (in angle).
    """
    if normed:
        xyz = xyz / np.sqrt(np.sum(xyz**2, axis=1, dtype=float, keepdims=True)).astype(np.float32)

    cos_sim = np.abs(np.dot(sphere.vertices, xyz.T))
    return np.argmax(cos_sim, axis=0)


def logsumexp(x, axis=None, keepdims=False):
    max_value = T.max(x, axis=axis, keepdims=True)
    res = max_value + T.log(T.sum(T.exp(x-max_value), axis=axis, keepdims=True))
    if not keepdims:
        if axis is None:
            return T.squeeze(res)

        slices = [slice(None, None, None)]*res.ndim
        slices[axis] = 0  # Axis being merged
        return res[tuple(slices)]

    return res


def softmax(x, axis=None):
    return T.exp(x - logsumexp(x, axis=axis, keepdims=True))


def chunk(sequence, n):
    """ Yield successive n-sized chunks from sequence. """
    for i in range(0, len(sequence), n):
        yield sequence[i:i + n]


def log_variables(batch_scheduler, model, *symb_vars):
    # Gather updates from the optimizer and the batch scheduler.
    f = theano.function([],
                        symb_vars,
                        givens=batch_scheduler.givens,
                        updates=model.updates,
                        name="compute_loss",
                        on_unused_input='ignore')

    log = [[] for _ in range(len(symb_vars))]
    for j in batch_scheduler:
        print(j)
        for i, e in enumerate(f()):
            log[i].append(e.copy())

    # return [list(itertools.chain(*l)) for l in log]
    return log  # [list(itertools.chain(*l)) for l in log]


def maybe_create_experiment_folder(args, exclude=[]):
    # Extract experiments hyperparameters
    hyperparams = OrderedDict(sorted(vars(args).items()))

    # Remove hyperparams that should not be part of the hash
    for name in exclude:
        del hyperparams[name]

    # Get/generate experiment name
    experiment_name = args.name
    if experiment_name is None:
        experiment_name = generate_uid_from_string(repr(hyperparams))

    # Create experiment folder
    experiment_path = pjoin(".", "experiments", experiment_name)
    resuming = False
    if os.path.isdir(experiment_path) and not args.force:
        resuming = True
        print("### Resuming experiment ({0}). ###\n".format(experiment_name))
        # Check if provided hyperparams match those in the experiment folder
        hyperparams_loaded = smartutils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))
        if hyperparams != hyperparams_loaded:
            print("{\n" + "\n".join(["{}: {}".format(k, hyperparams[k]) for k in sorted(hyperparams.keys())]) + "\n}")
            print("{\n" + "\n".join(["{}: {}".format(k, hyperparams_loaded[k]) for k in sorted(hyperparams_loaded.keys())]) + "\n}")
            print("The arguments provided are different than the one saved. Use --force if you are certain.\nQuitting.")
            sys.exit(1)
    else:
        if os.path.isdir(experiment_path):
            shutil.rmtree(experiment_path)

        os.makedirs(experiment_path)
        smartutils.save_dict_to_json_file(pjoin(experiment_path, "hyperparams.json"), hyperparams)

    return experiment_path, hyperparams, resuming
