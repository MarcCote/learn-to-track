from __future__ import division

import numpy as np

import theano
import theano.tensor as T
import nibabel as nib

from smartlearner import Dataset
from learn2track.utils import Timer
from learn2track import neurotools
from learn2track.neurotools import TractographyData

floatX = theano.config.floatX


class SequenceDataset(Dataset):
    """ Dataset interface.

    Attributes
    ----------
    symb_inputs : `theano.tensor.TensorType` object
        Symbolic variables representing the inputs.
    symb_targets : `theano.tensor.TensorType` object or None
        Symbolic variables representing the targets.

    Notes
    -----
    `symb_inputs` and `symb_targets` have test value already tagged to them. Use
    THEANO_FLAGS="compute_test_value=warn" to use them.
    """
    def __init__(self, inputs, targets=None, name="dataset", keep_on_cpu=False):
        """
        Parameters
        ----------
        inputs : list of ndarray
            Training examples (can be variable length sequences).
        targets : ndarray (optional)
            Target for each training example (can be variable length sequences).
        name : str (optional)
            The name of the dataset is used to name Theano variables. Default: 'dataset'.
        """
        self.keep_on_cpu = keep_on_cpu
        self.name = name
        self.inputs = inputs
        self.targets = targets

        self.symb_inputs = T.TensorVariable(type=T.TensorType("floatX", [False]*(inputs[0].ndim+1)),
                                            name=self.name+'_symb_inputs')
        self.symb_inputs.tag.test_value = inputs[0][None, ...]  # For debugging Theano graphs.

        self.symb_targets = None
        if self.has_targets:
            self.symb_targets = T.TensorVariable(type=T.TensorType("floatX", [False]*(targets[0].ndim+1)),
                                                 name=self.name+'_symb_targets')
            self.symb_targets.tag.test_value = targets[0][None, ...]  # For debugging Theano graphs.

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        if value is not None:
            self._targets = value
        else:
            self._targets = None

    @property
    def has_targets(self):
        return self.targets is not None

    @property
    def input_shape(self):
        return self.inputs[0].shape

    @property
    def target_shape(self):
        if self.has_targets:
            return self.targets[0].shape

        return None

    @property
    def input_size(self):
        # TODO: is this property really useful? If needed one could just call directly `dataset.input_shape[-1]`.
        return self.input_shape[-1]

    @property
    def target_size(self):
        # TODO: is this property really useful? If needed one could just call directly `dataset.target_shape[-1]`.
        if self.has_targets:
            return self.target_shape[-1]

        return None

    def __len__(self):
        return len(self.inputs)


class MaskedSequenceDataset(SequenceDataset):
    """ Dataset interface.

    Attributes
    ----------
    symb_inputs : `theano.tensor.TensorType` object
        Symbolic variables representing the inputs.
    symb_targets : `theano.tensor.TensorType` object or None
        Symbolic variables representing the targets.

    Notes
    -----
    `symb_inputs` and `symb_targets` have test value already tagged to them. Use
    THEANO_FLAGS="compute_test_value=warn" to use them.
    """
    def __init__(self, inputs, targets=None, name="dataset", keep_on_cpu=False):
        """
        Parameters
        ----------
        inputs : list of ndarray
            Training examples (can be variable length sequences).
        targets : ndarray (optional)
            Target for each training example (can be variable length sequences).
        name : str (optional)
            The name of the dataset is used to name Theano variables. Default: 'dataset'.
        """
        super().__init__(inputs, targets, name, keep_on_cpu)
        self.symb_mask = T.TensorVariable(type=T.TensorType("floatX", [False]*inputs[0].ndim),
                                          name=self.name+'_symb_mask')
        self.symb_mask.tag.test_value = (inputs[0][:, 0] > 0.5).astype(floatX)[None, ...]  # For debugging Theano graphs.


class TractographyDataset(MaskedSequenceDataset):
    def __init__(self, subjects, name="dataset", keep_on_cpu=False):
        """
        Parameters
        ----------
        subjects: list of TractogramData
        """
        self.subjects = subjects

        # Combine all tractograms in one.
        self.streamlines = nib.streamlines.ArraySequence()
        for i, subject in enumerate(self.subjects):
            self.streamlines.extend(subject.streamlines)

        super().__init__(self.streamlines, targets=None, name=name, keep_on_cpu=keep_on_cpu)

        # Build int2indices
        self.streamline_id_to_volume_id = np.nan * np.ones((len(self.streamlines),))

        start = 0
        for subject in self.subjects:
            end = start + len(subject.streamlines)
            self.streamline_id_to_volume_id[start:end] = subject.subject_id
            start = end

        assert not np.isnan(self.streamline_id_to_volume_id.sum())
        self.streamline_id_to_volume_id = self.streamline_id_to_volume_id.astype(floatX)

    def __len__(self):
        return len(self.streamlines)

    def __getitem__(self, idx):
        return self.streamlines[idx], self.streamline_id_to_volume_id[idx]


def load_tractography_dataset(subject_files, volume_manager, name="HCP", use_sh_coeffs=False):
    subjects = []
    with Timer("  Loading subject(s)", newline=True):
        for i, subject_file in enumerate(sorted(subject_files)):
            print("    {}".format(subject_file))
            tracto_data = TractographyData.load(subject_file)

            dwi = tracto_data.signal
            bvals = tracto_data.gradients.bvals
            bvecs = tracto_data.gradients.bvecs
            if use_sh_coeffs:
                # Use 45 spherical harmonic coefficients to represent the diffusion signal.
                volume = neurotools.get_spherical_harmonics_coefficients(dwi, bvals, bvecs, first=i==0).astype(np.float32)
            else:
                # Resample the diffusion signal to have 100 directions.
                volume = neurotools.resample_dwi(dwi, bvals, bvecs).astype(np.float32)

            tracto_data.signal.uncache()  # Free some memory as we don't need the original signal.
            subject_id = volume_manager.register(volume)
            tracto_data.subject_id = subject_id
            subjects.append(tracto_data)

    return TractographyDataset(subjects, name, keep_on_cpu=True)
