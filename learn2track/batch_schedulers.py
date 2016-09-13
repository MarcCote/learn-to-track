import theano
import numpy as np
import pickle
import theano.tensor as T
from os.path import join as pjoin

from dipy.tracking.streamline import set_number_of_points

from smartlearner.interfaces import BatchScheduler
from smartlearner.utils import sharedX

floatX = theano.config.floatX


class TractographyBatchScheduler(BatchScheduler):
    """ Batch scheduler for streamlines coming from multiple subjects. """
    def __init__(self, dataset, batch_size, noisy_streamlines_sigma=None, seed=1234, use_data_augment=True, normalize_target=False, shuffle_streamlines=True, resample_streamlines=True):
        """
        Parameters
        ----------
        dataset : :class:`TractographyDataset`
        batch_size : int
        use_data_augment : bool
            If true, perform data augmentation by flipping streamlines.
        shuffle_streamlines : bool
            Shuffle streamlines in the dataset between each epoch.
        resample_streamlines : bool
            Streamlines in a same batch will all have the same number of points.
            Should be always set to True for now (until the method _process_batch supports it).
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.use_augment_by_flipping = use_data_augment
        self.normalize_target = normalize_target

        self.noisy_streamlines_sigma = noisy_streamlines_sigma
        self.use_noisy_streamlines = self.noisy_streamlines_sigma is not None

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rng_noise = np.random.RandomState(self.seed+1)
        self.shuffle_streamlines = shuffle_streamlines
        self.resample_streamlines = resample_streamlines
        self.indices = np.arange(len(self.dataset))

        # Shared variables
        self._shared_batch_inputs = sharedX(np.ndarray((0, 0, 0)))
        self._shared_batch_targets = sharedX(np.ndarray((0, 0, 0)))
        self._shared_batch_mask = sharedX(np.ndarray((0, 0)))

        # Test value
        batch_inputs, batch_targets, batch_mask = self._next_batch(0)
        self.dataset.symb_inputs.tag.test_value = batch_inputs
        self.dataset.symb_mask.tag.test_value = batch_mask

        # Since this batch scheduler creates its own targets.
        if self.dataset.symb_targets is None:
            self.dataset.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False]*(batch_targets.ndim)),
                                                         name=self.dataset.name+'_symb_targets')

        self.dataset.symb_targets.tag.test_value = batch_targets

    @property
    def input_size(self):
        # Number of diffusion directions or Spherical Harmonics (SH) coefficients
        return self.dataset.volumes[0].shape[-1]

    @property
    def target_size(self):
        return 3  # Direction to follow.

    @property
    def nb_updates_per_epoch(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    def _augment_data(self, inputs):
        pass

    def _add_noise_to_streamlines(self, streamlines):
        if not self.use_noisy_streamlines:
            return streamlines

        # Add gaussian noise, N(0, self.sigma).
        noisy_streamlines = streamlines.copy()
        shape = noisy_streamlines._data.shape
        noisy_streamlines._data += self.noisy_streamlines_sigma * self.rng_noise.randn(*shape)
        return noisy_streamlines

    def _prepare_batch(self, indices):
        orig_streamlines, volume_ids = self.dataset[indices]
        streamlines = self._add_noise_to_streamlines(orig_streamlines.copy())

        streamlines._lengths = streamlines._lengths.astype("int64")
        if self.resample_streamlines:
            # streamline_length = np.max(streamlines._lengths)  # Sequences are resampled so that they have the same length.
            streamline_length = np.min(streamlines._lengths)  # Sequences are resampled so that they have the same length.
            streamlines = set_number_of_points(streamlines, nb_points=streamline_length)

        inputs = streamlines._data  # Streamlines coordinates
        targets = streamlines._data[1:] - streamlines._data[:-1]  # Unnormalized directions
        if self.normalize_target:
            targets = targets / np.sqrt(np.sum(targets**2, axis=1, keepdims=True))  # Normalized directions

        batch_size = len(streamlines)
        if self.use_augment_by_flipping:
            batch_size *= 2

        max_streamline_length = np.max(streamlines._lengths)  # Sequences are padded so that they have the same length.
        batch_masks = np.zeros((batch_size, max_streamline_length-1), dtype=floatX)
        batch_inputs = np.zeros((batch_size, max_streamline_length-1, inputs.shape[1]), dtype=floatX)
        batch_targets = np.zeros((batch_size, max_streamline_length-1, 3), dtype=floatX)

        for i, (offset, length, volume_id) in enumerate(zip(streamlines._offsets, streamlines._lengths, volume_ids)):
            batch_masks[i, :length-1] = 1
            batch_inputs[i, :length-1] = inputs[offset:offset+length-1]  # [0, 1, 2, 3, 4] => [0, 1, 2, 3]
            batch_targets[i, :length-1] = targets[offset:offset+length-1]  # [1-0, 2-1, 3-2, 4-3] => [1-0, 2-1, 3-2, 4-3]

            if self.use_augment_by_flipping:
                batch_masks[i+len(streamlines), :length-1] = 1
                batch_inputs[i+len(streamlines), :length-1] = inputs[offset+1:offset+length][::-1]  # [0, 1, 2, 3, 4] => [4, 3, 2, 1]
                batch_targets[i+len(streamlines), :length-1] = -targets[offset:offset+length-1][::-1]  # [1-0, 2-1, 3-2, 4-3] => [4-3, 3-2, 2-1, 1-0]

        batch_volume_ids = np.tile(volume_ids[:, None, None], (1 + self.use_augment_by_flipping, max_streamline_length-1, 1))
        batch_inputs = np.concatenate([batch_inputs, batch_volume_ids], axis=2)  # Streamlines coords + dwi ID
        return batch_inputs, batch_targets, batch_masks

    def _next_batch(self, batch_count):
        # Simply take the next slice.
        start = batch_count * self.batch_size
        end = (batch_count + 1) * self.batch_size
        return self._prepare_batch(self.indices[slice(start, end)])

    @property
    def givens(self):
        return {self.dataset.symb_inputs: self._shared_batch_inputs,
                self.dataset.symb_targets: self._shared_batch_targets,
                self.dataset.symb_mask: self._shared_batch_mask}

    def __iter__(self):
        if self.shuffle_streamlines:
            self.rng.shuffle(self.indices)

        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            yield batch_count + 1

    @property
    def updates(self):
        return {}  # No updates

    def save(self, savedir):
        state = {"version": 3,
                 "batch_size": self.batch_size,
                 "noisy_streamlines_sigma": self.noisy_streamlines_sigma,
                 "use_augment_by_flipping": self.use_augment_by_flipping,
                 "normalize_target": self.normalize_target,
                 "shuffle_streamlines": self.shuffle_streamlines,
                 "resample_streamlines": self.resample_streamlines,
                 "seed": self.seed,
                 "rng": pickle.dumps(self.rng),
                 "rng_noise": pickle.dumps(self.rng_noise),
                 "indices": self.indices,
                 }

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]
        self.noisy_streamlines_sigma = state["noisy_streamlines_sigma"]
        self.use_augment_by_flipping = state["use_augment_by_flipping"]
        self.rng = pickle.loads(state["rng"])
        self.rng_noise = pickle.loads(state["rng_noise"])
        self.indices = state["indices"]
        if state["version"] < 2:
            self.normalize_target = True
        else:
            self.normalize_target = state["normalize_target"]

        if state["version"] < 3:
            self.shuffle_streamlines = state["shuffle_streamlines"]
            self.resample_streamlines = state["resample_streamlines"]


class TractographyBatchSchedulerWithProportionalSamplingFromBundles(BatchScheduler):
    """ TODO

    Right now, there are only a few snippets of code coming from
    https://github.com/MarcCote/learn2track/blob/aa87673e76062eb0254339fe9f30500037ec35fc/learn2track/batch_schedulers.py#L16
    but that would certainly need to be adapted to handle multiple subjects.

    Some things to verify/do:
     - make sure the i-th bundle from subject correspond to the i-th in another.
       It should be the case if the datasets where generated with the script
       `process_streamlines.py`.
     - adapt the dataset so it is easier to sample streamlines from a given bundle
       using its ID (and that across all subjects).

    """

    @property
    def nb_updates_per_epoch(self):
        if self._nb_updates_per_epoch is None:
            return int(np.ceil(len(self.dataset) / self.batch_size))

        return self._nb_updates_per_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        # Compute the number of streamlines from each bundle that should be present in a batch.
        nb_bundles = int(np.sum(self.dataset.bundle_counts > 0))  # Keep only bundles that are present.
        nb_streamlines_from_each_bundle = (value//nb_bundles) * np.ones(nb_bundles, dtype=int)

        # Make sure the splits sum to `batch_size`.
        nb_streamlines_from_each_bundle[:(value % nb_bundles)] += 1

        self._nb_streamlines_from_each_bundle = np.zeros(len(self.dataset.bundle_names), dtype=int)
        self._nb_streamlines_from_each_bundle[self.dataset.bundle_counts > 0] = nb_streamlines_from_each_bundle
        assert sum(self._nb_streamlines_from_each_bundle) == self.batch_size

        # Pre-allocated memory for indices (speedup)
        self._indices = np.zeros((self._batch_size,), dtype=int)

    def _next_batch(self, batch_count):
        # Batch is a stratified sample of streamlines w.r.t. different bundles.
        start = 0
        for bundle_indices, nb_streamlines in zip(self.dataset.bundle_indices, self._nb_streamlines_from_each_bundle):
            if nb_streamlines == 0:
                continue

            end = start + nb_streamlines
            self._indices[start:end] = self.rng.choice(bundle_indices, size=(nb_streamlines,), replace=False)
            start = end

        return self._prepare_batch(self._indices)


class MultistepSequenceBatchScheduler(BatchScheduler):
    """ Multistep batch scheduler for streamlines dataset.

    TODO: this should probably inherit from TractographyBatchScheduler
    since there is a lot of similar code.
    """

    def __init__(self, dataset, batch_size, k, noisy_streamlines_sigma=None, nb_updates_per_epoch=None, seed=1234, include_last_point=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.k = k

        self.include_last_point = include_last_point

        self.use_augment_by_flipping = True

        self._nb_updates_per_epoch = nb_updates_per_epoch
        self.use_sample_from_bundle = self._nb_updates_per_epoch is not None

        self.noisy_streamlines_sigma = noisy_streamlines_sigma
        self.use_noisy_streamlines = self.noisy_streamlines_sigma is not None

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rng_noise = np.random.RandomState(self.seed + 1)

        # Shared variables
        self._shared_batch_inputs = sharedX(np.ndarray((0, 0, 0)))
        self._shared_batch_targets = sharedX(np.ndarray((0, 0, 0, 0)))
        self._shared_batch_mask = sharedX(np.ndarray((0, 0)))

        # Test value
        batch_inputs, batch_targets, batch_mask = self._next_batch(0)
        self.dataset.symb_inputs.tag.test_value = batch_inputs
        self.dataset.symb_mask.tag.test_value = batch_mask

        # Since this batch scheduler creates its own targets.
        if self.dataset.symb_targets is None:
            self.dataset.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False] * batch_targets.ndim),
                                                         name=self.dataset.name + '_symb_targets')

        self.dataset.symb_targets.tag.test_value = batch_targets


    @property
    def target_size(self):
        return 3  # Unnormalized direction vector

    @property
    def nb_updates_per_epoch(self):
        if self._nb_updates_per_epoch is None:
            return int(np.ceil(len(self.dataset) / self.batch_size))

        return self._nb_updates_per_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        # Compute the number of streamlines from each bundle that should be present in a batch.
        nb_bundles = int(np.sum(self.dataset.bundle_counts > 0))  # Keep only bundles that are present.
        nb_streamlines_from_each_bundle = (value // nb_bundles) * np.ones(nb_bundles, dtype=int)

        # Make sure the splits sum to `batch_size`.
        nb_streamlines_from_each_bundle[:(value % nb_bundles)] += 1

        self._nb_streamlines_from_each_bundle = np.zeros(len(self.dataset.bundle_names), dtype=int)
        self._nb_streamlines_from_each_bundle[self.dataset.bundle_counts > 0] = nb_streamlines_from_each_bundle
        assert sum(self._nb_streamlines_from_each_bundle) == self.batch_size

        # Pre-allocated memory for indices (speedup)
        self._indices = np.zeros((self._batch_size,), dtype=int)

    def _add_noise_to_streamlines(self, streamlines):
        if not self.use_noisy_streamlines:
            return streamlines

        # Add gaussian noise, N(0, self.sigma).
        noisy_streamlines = streamlines.copy()
        shape = noisy_streamlines._data.shape
        noisy_streamlines._data += self.noisy_streamlines_sigma * self.rng_noise.randn(*shape)
        return noisy_streamlines

    def _prepare_batch(self, indices):
        orig_streamlines = self.dataset.streamlines[indices].copy()
        streamlines = self._add_noise_to_streamlines(orig_streamlines)

        inputs = streamlines._data  # Streamlines coordinates
        targets = streamlines._data[1:] - streamlines._data[:-1]  # Unnormalized directions

        batch_size = len(streamlines)
        if self.use_augment_by_flipping:
            batch_size *= 2

        if self.include_last_point:  # only for the input
            raise NotImplementedError()

        else:
            max_streamline_length = np.max(streamlines._lengths)  # Sequences are padded so that they have the same length.
            batch_masks = np.zeros((batch_size, max_streamline_length - self.k), dtype=floatX)
            batch_inputs = np.zeros((batch_size, max_streamline_length - self.k, inputs.shape[1]), dtype=floatX)
            batch_targets = np.zeros((batch_size, max_streamline_length - self.k, self.k, self.target_size), dtype=floatX)

            for i, (offset, length) in enumerate(zip(streamlines._offsets, streamlines._lengths)):
                n = length - self.k
                batch_masks[i, :n] = 1
                batch_inputs[i, :n] = inputs[offset:offset + n]
                batch_targets[i, :n] = self._window_stack(targets[offset:offset + length - 1, None], self.k)

                if self.use_augment_by_flipping:
                    batch_masks[i + len(streamlines), :n] = 1
                    batch_inputs[i + len(streamlines), :n] = inputs[offset + self.k:offset + length][::-1]
                    batch_targets[i + len(streamlines), :n] = self._window_stack(-targets[offset:offset + length - 1, None][::-1], self.k)

        return batch_inputs, batch_targets, batch_masks

    @staticmethod
    def _window_stack(x, width):
        return np.hstack(x[i:1 + i - width or None] for i in range(width))

    def _next_batch(self, batch_count):
        if not self.use_sample_from_bundle:
            # Simply take the next slice.
            start = batch_count * self.batch_size
            end = (batch_count + 1) * self.batch_size
            return self._prepare_batch(slice(start, end))

        # Batch is a stratified sample of streamlines from the different bundles.
        start = 0
        for bundle_indices, nb_streamlines in zip(self.dataset.bundle_indices, self._nb_streamlines_from_each_bundle):
            if nb_streamlines == 0:
                continue

            end = start + nb_streamlines
            self._indices[start:end] = self.rng.choice(bundle_indices, size=(nb_streamlines,), replace=False)
            start = end

        return self._prepare_batch(self._indices)

    @property
    def givens(self):
        return {
            self.dataset.symb_inputs: self._shared_batch_inputs,
            self.dataset.symb_targets: self._shared_batch_targets,
            self.dataset.symb_mask: self._shared_batch_mask}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets, batch_mask = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)
            self._shared_batch_mask.set_value(batch_mask)

            yield batch_count + 1

    @property
    def updates(self):
        return {}  # No updates

    def save(self, savedir):
        state = {
            "version": 1,
            "batch_size": self.batch_size,
            "k": self.k,
            "nb_updates_per_epoch": self._nb_updates_per_epoch,
            "noisy_streamlines_sigma": self.noisy_streamlines_sigma,
            "use_augment_by_flipping": self.use_augment_by_flipping,
            "seed": self.seed,
            "rng": pickle.dumps(self.rng),
            "rng_noise": pickle.dumps(self.rng_noise)}

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]
        self.k = state["k"]
        self._nb_updates_per_epoch = state["nb_updates_per_epoch"]
        self.noisy_streamlines_sigma = state["noisy_streamlines_sigma"]
        self.use_augment_by_flipping = state["use_augment_by_flipping"]
        self.rng = pickle.loads(state["rng"])
        self.rng_noise = pickle.loads(state["rng_noise"])


class MultistepSequenceBatchSchedulerWithoutMask(BatchScheduler):
    """ Multistep batch scheduler for streamlines dataset.

    The only difference with MultistepSequenceBatchScheduler is that we
    assume (forces) all streamline to have the same number of points
    (same seq_length) for a given batch. To do so, we resample the streamlines
    so they have the same number of points as the shortest (arbitrary choice)
    streamline.
    """

    def __init__(self, dataset, batch_size, k, noisy_streamlines_sigma=None, nb_updates_per_epoch=None, seed=1234, include_last_point=False):
        self.dataset = dataset
        self.batch_size = batch_size

        self.k = k

        self.include_last_point = include_last_point

        self.use_augment_by_flipping = True

        self._nb_updates_per_epoch = nb_updates_per_epoch
        self.use_sample_from_bundle = self._nb_updates_per_epoch is not None

        self.noisy_streamlines_sigma = noisy_streamlines_sigma
        self.use_noisy_streamlines = self.noisy_streamlines_sigma is not None

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.rng_noise = np.random.RandomState(self.seed + 1)

        # No need for a mask since streamlines are going to be resampled.
        self.dataset.symb_mask = None

        # Shared variables
        self._shared_batch_inputs = sharedX(np.ndarray((0, 0, 0)))
        self._shared_batch_targets = sharedX(np.ndarray((0, 0, 0, 0)))

        # Test value
        batch_inputs, batch_targets = self._next_batch(0)
        self.dataset.symb_inputs.tag.test_value = batch_inputs

        # Since this batch scheduler creates its own targets.
        if self.dataset.symb_targets is None:
            self.dataset.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False] * batch_targets.ndim),
                                                         name=self.dataset.name + '_symb_targets')

        self.dataset.symb_targets.tag.test_value = batch_targets

    @property
    def input_size(self):
        return self.dataset.volume.shape[-1]  # Number of diffusion directions or coefficients of spherical harmonics

    @property
    def target_size(self):
        return 3  # Unnormalized direction vector

    @property
    def nb_updates_per_epoch(self):
        if self._nb_updates_per_epoch is None:
            return int(np.ceil(len(self.dataset) / self.batch_size))

        return self._nb_updates_per_epoch

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

        # Compute the number of streamlines from each bundle that should be present in a batch.
        nb_bundles = int(np.sum(self.dataset.bundle_counts > 0))  # Keep only bundles that are present.
        nb_streamlines_from_each_bundle = (value // nb_bundles) * np.ones(nb_bundles, dtype=int)

        # Make sure the splits sum to `batch_size`.
        nb_streamlines_from_each_bundle[:(value % nb_bundles)] += 1

        self._nb_streamlines_from_each_bundle = np.zeros(len(self.dataset.bundle_names), dtype=int)
        self._nb_streamlines_from_each_bundle[self.dataset.bundle_counts > 0] = nb_streamlines_from_each_bundle
        assert sum(self._nb_streamlines_from_each_bundle) == self.batch_size

        # Pre-allocated memory for indices (speedup)
        self._indices = np.zeros((self._batch_size,), dtype=int)

    def _add_noise_to_streamlines(self, streamlines):
        if not self.use_noisy_streamlines:
            return streamlines

        # Add gaussian noise, N(0, self.sigma).
        noisy_streamlines = streamlines.copy()
        shape = noisy_streamlines._data.shape
        noisy_streamlines._data += self.noisy_streamlines_sigma * self.rng_noise.randn(*shape)
        return noisy_streamlines

    def _prepare_batch(self, indices):
        orig_streamlines = self.dataset.streamlines[indices].copy()
        streamlines = self._add_noise_to_streamlines(orig_streamlines)

        # streamline_length = np.max(streamlines._lengths)  # Sequences are resampled so that they have the same length.
        streamline_length = np.min(streamlines._lengths)  # Sequences are resampled so that they have the same length.
        streamlines._lengths = streamlines._lengths.astype("int64")
        streamlines = set_number_of_points(streamlines, nb_points=streamline_length)
        inputs = streamlines._data  # Streamlines coordinates
        targets = streamlines._data[1:] - streamlines._data[:-1]  # Unnormalized directions

        batch_size = len(streamlines)
        if self.use_augment_by_flipping:
            batch_size *= 2

        if self.include_last_point:  # only for the input
            raise NotImplementedError()

        else:
            batch_inputs = np.zeros((batch_size, streamline_length - self.k, inputs.shape[1]), dtype=floatX)
            batch_targets = np.zeros((batch_size, streamline_length - self.k, self.k, self.target_size), dtype=floatX)

            for i, (offset, length) in enumerate(zip(streamlines._offsets, streamlines._lengths)):
                n = length - self.k
                batch_inputs[i, :n] = inputs[offset:offset + n]
                batch_targets[i, :n] = self._window_stack(targets[offset:offset + length - 1, None], self.k)

                if self.use_augment_by_flipping:
                    batch_inputs[i + len(streamlines), :n] = inputs[offset + self.k:offset + length][::-1]
                    batch_targets[i + len(streamlines), :n] = self._window_stack(-targets[offset:offset + length - 1, None][::-1], self.k)

        return batch_inputs, batch_targets

    @staticmethod
    def _window_stack(x, width):
        return np.hstack(x[i:1 + i - width or None] for i in range(width))

    def _next_batch(self, batch_count):
        if not self.use_sample_from_bundle:
            # Simply take the next slice.
            start = batch_count * self.batch_size
            end = (batch_count + 1) * self.batch_size
            return self._prepare_batch(slice(start, end))

        # Batch is a stratified sample of streamlines from the different bundles.
        start = 0
        for bundle_indices, nb_streamlines in zip(self.dataset.bundle_indices, self._nb_streamlines_from_each_bundle):
            if nb_streamlines == 0:
                continue

            end = start + nb_streamlines
            self._indices[start:end] = self.rng.choice(bundle_indices, size=(nb_streamlines,), replace=False)
            start = end

        return self._prepare_batch(self._indices)

    @property
    def givens(self):
        return {
            self.dataset.symb_inputs: self._shared_batch_inputs,
            self.dataset.symb_targets: self._shared_batch_targets}

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            batch_inputs, batch_targets = self._next_batch(batch_count)
            self._shared_batch_inputs.set_value(batch_inputs)
            self._shared_batch_targets.set_value(batch_targets)

            yield batch_count + 1

    @property
    def updates(self):
        return {}  # No updates

    def save(self, savedir):
        state = {
            "version": 1,
            "batch_size": self.batch_size,
            "k": self.k,
            "nb_updates_per_epoch": self._nb_updates_per_epoch,
            "noisy_streamlines_sigma": self.noisy_streamlines_sigma,
            "use_augment_by_flipping": self.use_augment_by_flipping,
            "seed": self.seed,
            "rng": pickle.dumps(self.rng),
            "rng_noise": pickle.dumps(self.rng_noise)}

        np.savez(pjoin(savedir, type(self).__name__ + '.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, type(self).__name__ + '.npz'))
        self.batch_size = state["batch_size"]
        self.k = state["k"]
        self._nb_updates_per_epoch = state["nb_updates_per_epoch"]
        self.noisy_streamlines_sigma = state["noisy_streamlines_sigma"]
        self.use_augment_by_flipping = state["use_augment_by_flipping"]
        self.rng = pickle.loads(state["rng"])
        self.rng_noise = pickle.loads(state["rng_noise"])
