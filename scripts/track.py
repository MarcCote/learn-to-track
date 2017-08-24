#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import os

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import argparse
from os.path import join as pjoin

import theano
import time

import dipy
import nibabel as nib
from nibabel.streamlines import Tractogram
from dipy.tracking.streamline import compress_streamlines

from smartlearner import views
from smartlearner import utils as smartutils

from learn2track import datasets
from learn2track.factories import loss_factory, batch_scheduler_factory
from learn2track.utils import Timer

from learn2track import neurotools

floatX = theano.config.floatX

# Constant
STOPPING_MASK =       int('00000001', 2)
STOPPING_LENGTH =     int('00000010', 2)
STOPPING_CURVATURE =  int('00000100', 2)
STOPPING_LIKELIHOOD = int('00001000', 2)


def build_argparser():
    DESCRIPTION = "Generate a tractogram from a LSTM model trained on ismrm2015 challenge data."
    p = argparse.ArgumentParser(description=DESCRIPTION)

    p.add_argument('name', type=str, help='name/path of the experiment.')
    p.add_argument('dwi', type=str, help="diffusion weighted images (.nii|.nii.gz).")
    p.add_argument('--out', type=str,
                   help="name of the output tractogram (.tck|.trk). Default: auto generate a meaningful name")
    p.add_argument('--prefix', type=str,
                   help="prefix to use for the name of the output tractogram, only if it is auto generated.")

    p.add_argument('--seeds', type=str, nargs="+", required=True,
                   help="use extermities of the streamlines in these tractograms (.trk|.tck) as seed points.")
    p.add_argument('--nb-seeds-per-voxel', type=int, default=1,
                   help="number of seeds per voxel, only if --seeds is a seeding mask (i.e. a nifti file). Default: 1")
    p.add_argument('--seeding-rng-seed', type=int, default=1234,
                   help="seed for the random generator responsible of generating the seeds in the voxels. Default: 1234")

    deviation_angle = p.add_mutually_exclusive_group()
    deviation_angle.add_argument('--theta', metavar='ANGLE', type=float,
                                 help="Maximum angle between 2 steps (in degree). [45]")
    deviation_angle.add_argument('--curvature', metavar='RADIUS', type=float,
                                 help='Minimum radius of curvature R in mm. Replaces --theta.')

    p.add_argument('--min-length', type=int, help="minimum length (in mm) for a streamline. Default: 20 mm", default=20)
    p.add_argument('--max-length', type=int, help="maximum length (in mm) for a streamline. Default: 200 mm", default=200)
    p.add_argument('--step-size', type=float, help="step size between two consecutive points in a streamlines (in mm). Default: use model's output as-is")
    p.add_argument('--mask', type=str,
                   help="if provided, streamlines will stop if going outside this mask (.nii|.nii.gz).")
    p.add_argument('--mask-threshold', type=float, default=0.05,
                   help="streamlines will be terminating if they pass through a voxel with a value from the mask lower than this value. Default: 0.05")

    p.add_argument('--filter-threshold', type=float,
                   help="If specified, only streamlines with a loss value lower than the specified value will be kept.")

    p.add_argument('--batch-size', type=int, help="number of streamlines to process at the same time. Default: the biggest possible")

    p.add_argument('--dilate-mask', action="store_true",
                   help="if specified, apply binary dilation on the tracking mask.")
    p.add_argument('--dilate-seeding-mask', action="store_true",
                   help="if specified, apply binary dilation on the seeding mask.")

    p.add_argument('--discard-stopped-by-curvature', action="store_true",
                   help='if specified, discard streamlines having a too high curvature (i.e. tracking stopped because of that).')

    pft = p.add_argument_group("Particle Filtering Tractography")
    pft.add_argument('--pft-nb-retry', type=int, default=0,
                     help="How many 'rescue' attempts to make (see PFT algorithm). Default: 0 i.e. do not use pft.")

    pft.add_argument('--pft-nb-backtrack-steps', type=int, default=1,
                     help="If --pft-nb-retry > 0, how many steps will be backtracked before attempting the 'rescue'. Default: 1.")

    # Deterministic sampling
    p.add_argument('--use-max-component', action="store_true",
                   help="if specified, generate streamlines by using maximum probability instead of sampling")

    # Custom tracking (with streamlines deflection)
    p.add_argument('--track-like-peter', action="store_true",
                   help="if specified, use a similar tracking approach as Peter.")

    # Flipping options
    # TODO: detect flipping directly from `dwi` file.
    p.add_argument('--flip-x', action="store_true",
                   help="if specified, prediction direction will be flip in X")
    p.add_argument('--flip-y', action="store_true",
                   help="if specified, prediction direction will be flip in Y")
    p.add_argument('--flip-z', action="store_true",
                   help="if specified, prediction direction will be flip in Z")

    deprecated = p.add_argument_group("Deprecated")
    deprecated.add_argument('--append-previous-direction', action="store_true",
                            help="(Deprecated) if specified, the target direction of the last timestep will be concatenated to the input of the current timestep. (0,0,0) will be used for the first timestep.")

    deprecated.add_argument('--backward-tracking-algo', type=int,
                            help="(Deprecated) if specified, both senses of the direction obtained from the seed point will be explored. Default: 0, (i.e. only do tracking one direction) ", default=0)

    # Optional parameters
    p.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    p.add_argument('-f', '--force', action='store_true', help='overwrite existing tractogram')

    return p


def is_flag_set(flags, ref_flag):
    """ Checks which flags have the `ref_flag` set. """
    return ((flags.astype(np.uint8) & ref_flag) >> np.log2(ref_flag).astype(np.uint8)).astype(bool)


def count_flags(flags, ref_flag):
    """ Counts how many flags have the `ref_flag` set. """
    return is_flag_set(flags, ref_flag).sum()


def compute_loss_errors(streamlines, model, hyperparams):
    # Create dummy dataset for these new streamlines.
    tracto_data = neurotools.TractographyData(None, None, None)
    tracto_data.add(streamlines, bundle_name="Generated")
    tracto_data.subject_id = 0
    dataset = datasets.TractographyDataset([tracto_data], "Generated", keep_on_cpu=True)

    # Override K for gru_multistep
    if 'k' in hyperparams:
        hyperparams['k'] = 1
    batch_scheduler = batch_scheduler_factory(hyperparams, dataset, train_mode=False, batch_size_override=1000, use_data_augment=False)
    loss = loss_factory(hyperparams, model, dataset)
    loss_view = views.LossView(loss=loss, batch_scheduler=batch_scheduler)
    return loss_view.losses.view()


def make_is_outside_mask(mask, affine, threshold=0):
    """ Makes a function that checks which streamlines have their last coordinates outside a mask.

    Parameters
    ----------
    mask : 3D array
        3D image defining a mask. The interior of the mask is defined by voxels with value higher or equal to `threshold`.
    affine : ndarray of shape (4, 4)
        Matrix representing the affine transformation that aligns streamlines coordinates on top of `mask`.
    threshold : float
        Voxels value higher or equal to this threshold are considered as part of the interior of the mask.

    Returns
    -------
    function
    """
    def _is_outside_mask(streamlines):
        """
        Parameters
        ----------
        streamlines : 3D array of shape (n_streamlines, n_points, 3)
            Streamlines information.

        Returns
        -------
        outside : 1D array of shape (n_streamlines,)
            Array telling whether a streamline last coordinate is outside the mask.
        """
        last_coordinates = streamlines[:, -1, :]
        mask_values = neurotools.map_coordinates_3d_4d(mask, last_coordinates, affine=affine, order=1)
        return mask_values < threshold

    return _is_outside_mask


def make_is_too_long(max_length):
    """ Makes a function that checks which streamlines are exceedingly long.

    Parameters
    ----------
    max_length : int
        Maximum number of points a streamline can have.

    Returns
    -------
    function

    Notes
    -----
    That's what she said!
    """

    def _is_too_long(streamlines):
        """
        Parameters
        ----------
        streamlines : 3D array of shape (n_streamlines, n_points, 3)
            Streamlines information.

        Returns
        -------
        too_long : 1D array of shape (n_streamlines,)
            Array telling wheter a streamline is too long or not.
        """
        # np.isfinite(streamlines[:, :, 0]) > max_length
        return np.asarray([streamlines.shape[1] > max_length] * len(streamlines))

    return _is_too_long


def make_is_too_curvy(max_theta):
    """ Makes a function that checks which streamlines are exceedingly curvy.

    Parameters
    ----------
    max_theta : float
        Maximum angle, in degree, two consecutive segments can have with each other.

    Returns
    -------
    function

    Notes
    -----
    That's what she said!
    """
    max_theta = np.deg2rad(max_theta)  # Internally use radian.

    def _is_too_curvy(streamlines):
        """
        Parameters
        ----------
        streamlines : 3D array of shape (n_streamlines, n_points, 3)
            Streamlines information.

        Returns
        -------
        too_long : 1D array of shape (n_streamlines,)
            Array telling wheter a streamline is too long or not.
        """
        if streamlines.shape[1] < 3:
            # Not enough segments to test curvature.
            return np.asarray([False] * len(streamlines))

        last_segments = streamlines[:, -1] - streamlines[:, -2]
        before_last_segments = streamlines[:, -2] - streamlines[:, -3]

        # Normalized segments.
        last_segments /= np.sqrt(np.sum(last_segments**2, axis=1, keepdims=True))
        before_last_segments /= np.sqrt(np.sum(before_last_segments**2, axis=1, keepdims=True))

        # Compute angles.
        angles = np.arccos(np.sum(last_segments * before_last_segments, axis=1))
        return angles > max_theta

    return _is_too_curvy


def make_is_stopping(stopping_criteria):
    """ Makes a function that checks which streamlines should we stop tracking.

    Parameters
    ----------
    stopping_criteria : dict
        Dictionnary containing all stopping criteria to check. The key is one the
        flags constant variable defined above. The value can be any function
        that expect `streamlines` as input and returns a boolean array indicating
        which streamlines should be stopped..
    """

    def _is_stopping(streamlines, to_check=None):
        """
        Parameters
        ----------
        streamlines : 3D array of shape (n_streamlines, n_points, 3)
            Streamlines coordinates.
        to_check : 1D array, optional
            Only check specific streamlines.
            If array of bool, check only streamlines marked as True.
            If array of int, check only streamlines at specified index.
            By default, check every coordinate.

        Returns
        -------
        undone : 1D array
            Array containing the indices of ongoing streamlines.
        done : 1D array
            Array containing the indices of streamlines that should be stopped.
        flags : 1D array
            Array containing a flag explaining why a streamline should be stopped.
        """

        if to_check is None:
            idx = np.arange(len(streamlines))
        elif isinstance(to_check, np.ndarray) and to_check.dtype == np.bool:
            assert len(to_check) == len(streamlines)
            idx = np.where(to_check)[0]
        else:
            idx = to_check

        undone = np.ones(len(idx), dtype=bool)
        flags = np.zeros(len(idx), dtype=np.uint8)
        for flag, stopping_criterion in stopping_criteria.items():
            done = stopping_criterion(streamlines[idx])
            undone[done] = False
            flags[done] |= flag

        done = np.logical_not(undone)
        return idx[np.where(undone)[0]], idx[np.where(done)[0]], flags[done]

    return _is_stopping


def rotate(directions, axis, degree=180):
    assert degree == 180, "Only supports rotation of 180 degrees."
    new_directions = []
    for a, b in zip(directions, axis):
        new_directions += [-a + 2 * np.dot(a, b) / np.dot(b, b) * b]

    return np.stack(new_directions, axis=0)


class Tracker(object):
    def __init__(self, model, is_stopping, keep_last_n_states=1, use_max_component=False, flip_x=False, flip_y=False, flip_z=False, compress_streamlines=False):
        self.model = model
        self._is_stopping = is_stopping
        self.grower = model.make_sequence_generator(use_max_component=use_max_component)
        self.keep_last_n_states = max(keep_last_n_states, 1)
        self._history = []
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z
        self.compress_streamlines = compress_streamlines

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, values):
        # Add new states to history; free space if needed.
        self._history += [self._states.copy()]
        if len(self._history) > self.keep_last_n_states:
            self._history = self._history[1:]

        self._states = values.copy()

    def is_stopping(self, sprouts):
        undone, done, stopping_flags = self._is_stopping(sprouts)
        return undone, done, stopping_flags

    def is_ripe(self):
        return len(self.sprouts) == 0

    def get(self, flag):
        _, done, stopping_flags = self.is_stopping(self.sprouts)
        return done[(stopping_flags & flag) != 0]

    def plant(self, seeds):
        if seeds.ndim == 2:
            seeds = seeds[:, None, :]

        self.sprouts = seeds.copy()
        self._states = self.model.get_init_states(batch_size=len(seeds))

    def _grow_step(self, sprouts, states, step_size):

        # Always feed previous direction, grower will choose to use it or not
        if sprouts.shape[1] >= 2:
            previous_direction = sprouts[:, -1, :] - sprouts[:, -2, :]
        else:
            previous_direction = np.zeros_like(sprouts[:, -1, :])

        previous_direction = previous_direction / np.sqrt(np.sum(previous_direction ** 2, axis=1, keepdims=True) + 1e-6)

        # Get next unnormalized directions
        directions, new_states = self.grower(x_t=sprouts[:, -1, :], states=states, previous_direction=previous_direction)
        if self.flip_x:
            directions[:, 0] *= -1
        if self.flip_y:
            directions[:, 1] *= -1
        if self.flip_z:
            directions[:, 2] *= -1

        if step_size is not None:
            # Norm direction and stretch it.
            normalized_directions = directions / np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
            directions = normalized_directions * step_size

        # Take a step i.e. it's growing!
        new_sprouts = np.concatenate([sprouts, sprouts[:, [-1], :] + directions[:, None, :]], axis=1)
        return new_sprouts, new_states

    def grow(self, step_size):
        self.sprouts, self.states = self._grow_step(self.sprouts, self.states, step_size)

    def _keep(self, idx):
        # Update remaining sprouts and their states.
        self.sprouts = self.sprouts[idx]
        self._states = [s[idx] for s in self._states]

        # Rewrite history
        for i, state in enumerate(self._history):
            self._history[i] = [s[idx] for s in state]

    def harvest(self):
        undone, done, stopping_flags = self.is_stopping(self.sprouts)

        # Do not keep last point since it almost surely raised the stopping flag.
        streamlines = list(self.sprouts[done, :-1])
        if self.compress_streamlines:
            streamlines = compress_streamlines(streamlines)

        tractogram = Tractogram(streamlines=streamlines,
                                data_per_streamline={"stopping_flags": stopping_flags})

        # Keep only undone sprouts
        self._keep(undone)

        return tractogram

    def regrow(self, idx, step_size, backtrack_n_steps):
        if self.sprouts.shape[1] <= backtrack_n_steps:
            # Cannot regrow sprouts that are too small.
            return 0

        # Get sprouts that needs regrowing.
        sprouts = self.sprouts[idx, :-backtrack_n_steps]
        states = [s[idx] for s in self._history[-backtrack_n_steps]]
        idx_to_keep = np.arange(len(sprouts))

        local_history = []
        for _ in range(backtrack_n_steps):
            if len(sprouts) == 0:
                # Nothing left to regrow, no sprouts could be saved.
                return 0

            local_history += [states]
            sprouts, states = self._grow_step(sprouts, states, step_size)

            undone, _, _ = self.is_stopping(sprouts)
            sprouts = sprouts[undone]
            states = [s[undone] for s in states]
            idx_to_keep = idx_to_keep[undone]

            # Rewrite local history
            for i, old_states in enumerate(local_history):
                local_history[i] = [s[undone] for s in old_states]

        # Update original sprouts and their states.
        self.sprouts[idx[idx_to_keep]] = sprouts
        for i, state in enumerate(self._states):
            self._states[i][idx[idx_to_keep]] = states[i]

        # Rewrite history
        assert len(self._history[-backtrack_n_steps:]) == len(local_history)
        for j, old_states in enumerate(self._history[-backtrack_n_steps:], start=-backtrack_n_steps):
            for i, old_state in enumerate(old_states):
                self._history[j][i][idx[idx_to_keep]] = local_history[j][i]

        return len(idx_to_keep)  # Number of successful regrowths.


class PeterTracker(Tracker):

    def regrow(self, idx, step_size, backtrack_n_steps):
        # Try deflecting the streamline by rotating 180 degree around
        # the previous direction.
        assert backtrack_n_steps == 1, "Only last step can be deflected."

        if self.sprouts.shape[1] <= backtrack_n_steps + 1:
            # Cannot regrow sprouts that are too small.
            return 0

        # Get sprouts that needs regrowing.
        sprouts = self.sprouts[idx, :-backtrack_n_steps]
        idx_to_keep = np.arange(len(sprouts))

        if len(sprouts) == 0:
            # Nothing left to regrow, no sprouts could be saved.
            return 0

        previous_directions = self.sprouts[idx, -2, :] - self.sprouts[idx, -3, :]
        predicted_directions = self.sprouts[idx, -1, :] - self.sprouts[idx, -2, :]
        directions = rotate(predicted_directions, axis=previous_directions, degree=180)

        sprouts = np.concatenate([sprouts, sprouts[:, [-1], :] + directions[:, None, :]], axis=1)
        # TODO: need to update the state of RNN-like models.
        states = self.states
        # sprouts, states = self._grow_step(sprouts, states, step_size)

        undone, _, _ = self.is_stopping(sprouts)
        sprouts = sprouts[undone]
        states = [s[undone] for s in states]
        idx_to_keep = idx_to_keep[undone]

        # Update original sprouts and their states.
        self.sprouts[idx[idx_to_keep]] = sprouts
        for i, state in enumerate(self._states):
            self._states[i][idx[idx_to_keep]] = states[i]

        return len(idx_to_keep)  # Number of successful regrowths.


class BackwardTracker(Tracker):
    def plant(self, seeds):
        self.seeds = seeds
        self.nb_init_steps = np.asarray(list(map(len, seeds)))
        self.sprouts = np.asarray([s[0] for s in seeds])[:, None, :]
        self._states = self.model.get_init_states(batch_size=len(seeds))

    def is_stopping(self, sprouts):
        undone, done, stopping_flags = self._is_stopping(sprouts)

        # Ignore sprouts that haven't finished initializing.
        init_undone = self.nb_init_steps >= self.sprouts.shape[1]
        undone = np.r_[undone, [idx for idx in done if init_undone[idx]]]
        undone = undone.astype(int)

        # Sprouts can be done only if it has been initialized completely.
        truely_done = np.logical_not(init_undone[done])
        done = done[truely_done]
        stopping_flags = stopping_flags[truely_done]

        return undone, done, stopping_flags

    def grow(self, step_size):
        new_sprouts, self.states = self._grow_step(self.sprouts, self.states, step_size)

        # Only update sprouts once they are done initializing.
        # However always update their states.
        init_undone = self.nb_init_steps >= new_sprouts.shape[1]
        if np.any(init_undone):
            new_sprouts[init_undone, -1] = [s[new_sprouts.shape[1]-1] for s, undone in zip(self.seeds, init_undone) if undone]

        self.sprouts = new_sprouts

    def regrow(self, idx, step_size, backtrack_n_steps):
        init_done = self.nb_init_steps < self.sprouts.shape[1]
        assert np.all(init_done[idx])  # Call an exterminator if that happens!
        return super().regrow(idx, step_size, backtrack_n_steps)

    def _keep(self, idx):
        super()._keep(idx)
        self.seeds = [self.seeds[i] for i in np.arange(len(self.seeds))[idx]]
        self.nb_init_steps = self.nb_init_steps[idx]


class BackwardPeterTracker(BackwardTracker, PeterTracker):

    def regrow(self, idx, step_size, backtrack_n_steps):
        init_done = self.nb_init_steps < self.sprouts.shape[1]
        assert np.all(init_done[idx])  # Call an exterminator if that happens!
        return PeterTracker.regrow(self, idx, step_size, backtrack_n_steps)


def track(tracker, seeds, step_size, is_stopping, nb_retry=0, nb_backtrack_steps=0, verbose=False):
    """ Generates streamlines using the Particle Filtering Tractography algorithm.

    This algorithm is inspired from Girard etal. (2014) Neuroimage.

    Parameters
    ----------
    is_stopping : function
        Tells whether a streamlines should stop being tracked.
        This function expects a list of streamlines and returns a 3-tuple:
          the indices of the streamlines that are undone,
          the indices of the streamlines that are done,
          the reasons why the streamlines should be stopped.
    """
    tractogram = None
    tracker.plant(seeds)

    i = 1
    while not tracker.is_ripe():
        if verbose:
            print("pts: {}/{} ({:,} remaining)".format(i+1, is_stopping.max_nb_points, len(tracker.sprouts)), end="")

        tracker.grow(step_size)

        for _ in range(nb_retry):
            if verbose:
                print(".", end="")
                sys.stdout.flush()

            for backtrack_n_steps in range(1, nb_backtrack_steps+1):
                idx = tracker.get(flag=STOPPING_MASK | STOPPING_CURVATURE | STOPPING_LIKELIHOOD)
                if len(idx) == 0:
                    # No sprouts to be saved.
                    break

                nb_saved = tracker.regrow(idx, step_size, backtrack_n_steps=backtrack_n_steps)
                print("{}/{} saved.".format(nb_saved, len(idx)))

            if len(idx) == 0:
                # No sprouts to be saved.
                break

        if tractogram is None:
            tractogram = tracker.harvest()
        else:
            tractogram += tracker.harvest()

        if verbose and nb_retry == 0:
            print("")

        i += 1

    return tractogram


def batch_track(model, dwi, seeds, step_size, batch_size, is_stopping, args):
    if batch_size is None:
        batch_size = len(seeds)

    nb_retry = 1 if args.track_like_peter else args.pft_nb_retry
    nb_backtrack_steps = 1 if args.track_like_peter else args.pft_nb_backtrack_steps
    TrackerCls = PeterTracker if args.track_like_peter else Tracker
    BackwardTrackerCls = BackwardPeterTracker if args.track_like_peter else BackwardTracker

    while True:
        try:
            time.sleep(1)
            print("Trying to track {:,} streamlines at the same time.".format(batch_size))
            tractogram = None#nib.streamlines.Tractogram()

            for start in range(0, len(seeds), batch_size):
                print("{:,} / {:,}".format(start, len(seeds)))
                end = start + batch_size

                # Forward tracking
                tracker = TrackerCls(model, is_stopping, args.pft_nb_backtrack_steps, args.use_max_component,
                                     args.flip_x, args.flip_y, args.flip_z, compress_streamlines=False)
                batch_tractogram = track(tracker=tracker, seeds=seeds[start:end], step_size=step_size, is_stopping=is_stopping,
                                         nb_retry=nb_retry, nb_backtrack_steps=nb_backtrack_steps, verbose=args.verbose)

                stopping_flags = batch_tractogram.data_per_streamline['stopping_flags'].astype(np.uint8)
                print("Forward pass stopped because of - mask: {:,}\t curv: {:,}\t length: {:,}".format(count_flags(stopping_flags, STOPPING_MASK),
                                                                                                        count_flags(stopping_flags, STOPPING_CURVATURE),
                                                                                                        count_flags(stopping_flags, STOPPING_LENGTH)))

                # Backward tracking
                tracker = BackwardTrackerCls(model, is_stopping, args.pft_nb_backtrack_steps, args.use_max_component,
                                             args.flip_x, args.flip_y, args.flip_z, compress_streamlines=True)
                streamlines = [s[::-1] for s in batch_tractogram.streamlines]  # Flip streamlines (the first half).
                batch_tractogram = track(tracker=tracker, seeds=streamlines, step_size=step_size, is_stopping=is_stopping,
                                         nb_retry=nb_retry, nb_backtrack_steps=nb_backtrack_steps, verbose=args.verbose)

                stopping_flags = batch_tractogram.data_per_streamline['stopping_flags'].astype(np.uint8)
                print("Backward pass stopped because of - mask: {:,}\t curv: {:,}\t length: {:,}".format(count_flags(stopping_flags, STOPPING_MASK),
                                                                                                         count_flags(stopping_flags, STOPPING_CURVATURE),
                                                                                                         count_flags(stopping_flags, STOPPING_LENGTH)))

                if tractogram is None:
                    tractogram = batch_tractogram
                else:
                    tractogram += batch_tractogram

            return tractogram

        except MemoryError:
            print("{:,} streamlines is too much!".format(batch_size))
            batch_size //= 2
            if batch_size < 0:
                raise MemoryError("Might needs a bigger graphic card!")

        except RuntimeError as e:
            if "out of memory" in e.args[0]:
                print("{:,} streamlines is too much!".format(batch_size))
                batch_size //= 2
                if batch_size < 0:
                    raise MemoryError("Might needs a bigger graphic card!")

            else:
                raise e


def get_max_angle_from_curvature(curvature, step_size):
    """
    Parameters
    ----------
    curvature: float
        Minimum radius of curvature in mm.
    step_size: float
        The tracking step size in mm.

    Return
    ------
    theta: float
        The maximum deviation angle in radian,
        given the radius curvature and the step size.
    """
    theta = 2. * np.arcsin(step_size / (2. * curvature))
    if np.isnan(theta) or theta > np.pi / 2 or theta <= 0:
        theta = np.pi / 2.0
    return theta


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Get experiment folder
    experiment_path = args.name
    if not os.path.isdir(experiment_path):
        # If not a directory, it must be the name of the experiment.
        experiment_path = pjoin(".", "experiments", args.name)

    if not os.path.isdir(experiment_path):
        parser.error('Cannot find experiment: {0}!'.format(args.name))

    # Load experiments hyperparameters
    try:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(experiment_path, "hyperparams.json"))
    except FileNotFoundError:
        hyperparams = smartutils.load_dict_from_json_file(pjoin(experiment_path, "..", "hyperparams.json"))

    with Timer("Loading DWIs"):
        # Load gradients table
        dwi_name = args.dwi
        if dwi_name.endswith(".gz"):
            dwi_name = dwi_name[:-3]
        if dwi_name.endswith(".nii"):
            dwi_name = dwi_name[:-4]
        bvals_filename = dwi_name + ".bvals"
        bvecs_filename = dwi_name + ".bvecs"
        bvals, bvecs = dipy.io.gradients.read_bvals_bvecs(bvals_filename, bvecs_filename)

        dwi = nib.load(args.dwi)
        if hyperparams["use_sh_coeffs"]:
            # Use 45 spherical harmonic coefficients to represent the diffusion signal.
            weights = neurotools.get_spherical_harmonics_coefficients(dwi, bvals, bvecs).astype(np.float32)
        else:
            # Resample the diffusion signal to have 100 directions.
            weights = neurotools.resample_dwi(dwi, bvals, bvecs).astype(np.float32)

        affine_rasmm2dwivox = np.linalg.inv(dwi.affine)

    with Timer("Loading model"):
        if hyperparams["model"] == "gru_regression":
            from learn2track.models import GRU_Regression
            model_class = GRU_Regression
        elif hyperparams['model'] == 'gru_gaussian':
            from learn2track.models import GRU_Gaussian
            model_class = GRU_Gaussian
        elif hyperparams['model'] == 'gru_mixture':
            from learn2track.models import GRU_Mixture
            model_class = GRU_Mixture
        elif hyperparams['model'] == 'gru_multistep':
            from learn2track.models import GRU_Multistep_Gaussian
            model_class = GRU_Multistep_Gaussian
        elif hyperparams['model'] == 'ffnn_regression':
            from learn2track.models import FFNN_Regression
            model_class = FFNN_Regression
        else:
            raise ValueError("Unknown model!")

        kwargs = {}
        volume_manager = neurotools.VolumeManager()
        volume_manager.register(weights)
        kwargs['volume_manager'] = volume_manager

        # Load the actual model.
        model = model_class.create(pjoin(experiment_path), **kwargs)  # Create new instance and restore model.
        model.drop_prob = 0.
        print(str(model))

    mask = None
    if args.mask is not None:
        with Timer("Loading mask"):
            mask_nii = nib.load(args.mask)
            mask = mask_nii.get_data()
            # Compute the affine allowing to evaluate the mask at some coordinates correctly.

            # affine_maskvox2dwivox = mask_vox => rasmm space => dwi_vox
            affine_maskvox2dwivox = np.dot(affine_rasmm2dwivox, mask_nii.affine)
            if args.dilate_mask:
                import scipy
                mask = scipy.ndimage.morphology.binary_dilation(mask).astype(mask.dtype)

    with Timer("Generating seeds"):
        seeds = []

        for filename in args.seeds:
            if filename.endswith('.trk') or filename.endswith('.tck'):
                tfile = nib.streamlines.load(filename)
                # Send the streamlines to voxel since that's where we'll track.
                tfile.tractogram.apply_affine(affine_rasmm2dwivox)

                # Use extremities of the streamlines as seeding points.
                seeds += [s[0] for s in tfile.streamlines]
                seeds += [s[-1] for s in tfile.streamlines]

            else:
                # Assume it is a binary mask.
                rng = np.random.RandomState(args.seeding_rng_seed)
                nii_seeds = nib.load(filename)

                # affine_seedsvox2dwivox = mask_vox => rasmm space => dwi_vox
                affine_seedsvox2dwivox = np.dot(affine_rasmm2dwivox, nii_seeds.affine)

                nii_seeds_data = nii_seeds.get_data()

                if args.dilate_seeding_mask:
                    import scipy
                    nii_seeds_data = scipy.ndimage.morphology.binary_dilation(nii_seeds_data).astype(nii_seeds_data.dtype)
                    
                indices = np.array(np.where(nii_seeds_data)).T
                for idx in indices:
                    seeds_in_voxel = idx + rng.uniform(-0.5, 0.5, size=(args.nb_seeds_per_voxel, 3))
                    seeds_in_voxel = nib.affines.apply_affine(affine_seedsvox2dwivox, seeds_in_voxel)
                    seeds.extend(seeds_in_voxel)

        seeds = np.array(seeds, dtype=theano.config.floatX)

    with Timer("Tracking in the diffusion voxel space"):
        voxel_sizes = np.asarray(dwi.header.get_zooms()[:3])
        if not np.all(voxel_sizes == dwi.header.get_zooms()[0]):
            print("* Careful voxel are anisotropic {}!".format(tuple(voxel_sizes)))
        # Since we are tracking in diffusion voxel space, convert step_size (in mm) to voxel.

        if args.step_size is not None:
            step_size = np.float32(args.step_size / voxel_sizes.max())
            # Also convert max length (in mm) to voxel.
            max_nb_points = int(np.ceil(args.max_length / args.step_size))
        else:
            step_size = None
            max_nb_points = args.max_length

        if args.theta is not None:
            theta = np.deg2rad(args.theta)
        elif args.curvature is not None and args.curvature > 0:
            theta = get_max_angle_from_curvature(args.curvature, step_size)
        else:
            theta = np.deg2rad(45)

        print("Angle: {}".format(np.rad2deg(theta)))
        print("Step size (vox): {}".format(step_size))
        print("Max nb. points: {}".format(max_nb_points))

        is_outside_mask = make_is_outside_mask(mask, affine_maskvox2dwivox, threshold=args.mask_threshold)
        is_too_long = make_is_too_long(max_nb_points)
        is_too_curvy = make_is_too_curvy(np.rad2deg(theta))
        is_stopping = make_is_stopping({STOPPING_MASK: is_outside_mask,
                                        STOPPING_LENGTH: is_too_long,
                                        STOPPING_CURVATURE: is_too_curvy})

        is_stopping.max_nb_points = max_nb_points  # Small hack

        tractogram = batch_track(model, weights, seeds,
                                 step_size=step_size,
                                 is_stopping=is_stopping,
                                 batch_size=args.batch_size,
                                 args=args)

        # Streamlines have been generated in voxel space.
        # Transform them them back to RAS+mm space using the dwi's affine.
        tractogram.affine_to_rasmm = dwi.affine
        tractogram.to_world()  # Performed in-place.

    nb_streamlines = len(tractogram)
    print("Generated {:,} (compressed) streamlines".format(nb_streamlines))
    with Timer("Cleaning streamlines", newline=True):
        # Flush streamlines that has no points.
        tractogram = tractogram[np.array(list(map(len, tractogram))) > 0]
        print("Removed {:,} empty streamlines".format(nb_streamlines - len(tractogram)))

        # Remove small streamlines
        nb_streamlines = len(tractogram)
        lengths = dipy.tracking.streamline.length(tractogram.streamlines)
        tractogram = tractogram[lengths >= args.min_length]
        lengths = lengths[lengths >= args.min_length]
        if len(lengths) > 0:
            print("Average length: {:.2f} mm.".format(lengths.mean()))
            print("Minimum length: {:.2f} mm. Maximum length: {:.2f}".format(lengths.min(), lengths.max()))
        print("Removed {:,} streamlines smaller than {:.2f} mm".format(nb_streamlines - len(tractogram),
                                                                       args.min_length))
        if args.discard_stopped_by_curvature:
            nb_streamlines = len(tractogram)
            stopping_curvature_flag_is_set = is_flag_set(tractogram.data_per_streamline['stopping_flags'][:, 0], STOPPING_CURVATURE)
            tractogram = tractogram[np.logical_not(stopping_curvature_flag_is_set)]
            print("Removed {:,} streamlines stopped for having a curvature higher than {:.2f} degree".format(nb_streamlines - len(tractogram),
                                                                                                             np.rad2deg(theta)))

        if args.filter_threshold is not None:
            # Remove streamlines that produces a reconstruction error higher than a certain threshold.
            nb_streamlines = len(tractogram)
            losses = compute_loss_errors(tractogram.streamlines, model, hyperparams)
            print("Mean loss: {:.4f} ± {:.4f}".format(np.mean(losses), np.std(losses, ddof=1) / np.sqrt(len(losses))))
            tractogram = tractogram[losses <= args.filter_threshold]
            print("Removed {:,} streamlines producing a loss lower than {:.2f} mm".format(nb_streamlines - len(tractogram),
                                                                                          args.filter_threshold))

    with Timer("Saving {:,} (compressed) streamlines".format(len(tractogram))):
        filename = args.out
        if args.out is None:
            prefix = args.prefix
            if prefix is None:
                dwi_name = os.path.basename(args.dwi)
                if dwi_name.endswith(".nii.gz"):
                    dwi_name = dwi_name[:-7]
                else:  # .nii
                    dwi_name = dwi_name[:-4]

                prefix = os.path.basename(os.path.dirname(args.dwi)) + dwi_name
                prefix = prefix.replace(".", "_")

            seed_mask_type = args.seeds[0].replace(".", "_").replace("_", "").replace("/", "-")
            if "int" in args.seeds[0]:
                seed_mask_type = "int"
            elif "wm" in args.seeds[0]:
                seed_mask_type = "wm"
            elif "rois" in args.seeds[0]:
                seed_mask_type = "rois"
            elif "bundles" in args.seeds[0]:
                seed_mask_type = "bundles"
            
            if "fa" in args.mask:
                mask_type = "fa"
            elif "wm" in args.mask:
                mask_type = "wm"

            if args.dilate_seeding_mask:
                seed_mask_type += "D"

            if args.dilate_mask:
                mask_type += "D"

            filename_items = ["{}_",
                              # "seed-{}_",
                              # "mask-{}_",
                              "step-{:.2f}mm_",
                              "nbSeeds-{}_",
                              "maxAngleDeg-{:.1f}_"
                              # "keepCurv-{}_",
                              # "filtered-{}_",
                              # "minLen-{}_",
                              # "pftRetry-{}_",
                              # "pftHist-{}_",
                              # "trackLikePeter-{}_",
                              # "useMaxComponent-{}"
                              ]
            filename = ('_'.join(filename_items) + ".tck").format(
                prefix,
                # seed_mask_type,
                # mask_type,
                args.step_size,
                args.nb_seeds_per_voxel,
                np.rad2deg(theta)
                # not args.discard_stopped_by_curvature,
                # args.filter_threshold,
                # args.min_length,
                # args.pft_nb_retry,
                # args.pft_nb_backtrack_steps,
                # args.track_like_peter,
                # args.use_max_component
                )

        save_path = pjoin(experiment_path, filename)
        try:  # Create dirs, if needed.
            os.makedirs(os.path.dirname(save_path))
        except:
            pass

        print("Saving to {}".format(save_path))
        nib.streamlines.save(tractogram, save_path)

if __name__ == "__main__":
    main()
