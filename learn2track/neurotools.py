from collections import OrderedDict

import nibabel as nib
import numpy as np
import theano
import theano.tensor as T
from dipy.align.bundlemin import distance_matrix_mdf
from dipy.core.sphere import Sphere, HemiSphere
from dipy.data import get_sphere
from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from dipy.segment.quickbundles import QuickBundles
from dipy.tracking.streamline import set_number_of_points
from scipy.ndimage import map_coordinates
from smartlearner.utils import sharedX

from learn2track.interpolation import eval_volume_at_3d_coordinates_in_theano

floatX = theano.config.floatX


class TractographyData(object):
    def __init__(self, signal, gradients, name2id=None):
        """
        Parameters
        ----------
        signal : :class:`nibabel.Nifti1Image` object
            Diffusion signal used to generate the streamlines.
        gradients : :class:`dipy.core.gradients.GradientTable` object
            Diffusion gradient information for the `signal`.
        """
        self.streamlines = nib.streamlines.ArraySequence()
        self.bundle_ids = np.zeros((0,), dtype=np.int16)
        self.name2id = OrderedDict() if name2id is None else name2id
        self.signal = signal
        self.gradients = gradients
        self.subject_id = None
        self.filename = None

    @property
    def volume(self):
        if self._volume is None:
            # Returns original signal
            return self.signal.get_data()

        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def bundle_names(self):
        return list(self.name2id.keys())

    def add(self, streamlines, bundle_name=None, bundle_ids=None):
        """ Adds a bundle of streamlines to this container.

        Parameters
        ----------
        streamlines : `ArraySequence` object or list of 3D arrays
            Streamlines to be added.
        bundle_name : str
            Name of the bundle the streamlines belong to.
        """
        # Get bundle ID, create one if it's new bundle.
        if bundle_name is not None:
            if bundle_name not in self.name2id:
                self.name2id[bundle_name] = len(self.name2id)

            bundle_id = self.name2id[bundle_name]

        if bundle_ids is not None:
            bundle_id = bundle_ids

        # Append streamlines
        self.streamlines.extend(streamlines)
        size = len(self.bundle_ids)
        new_size = size + len(streamlines)
        self.bundle_ids.resize((new_size,))
        self.bundle_ids[size:new_size] = bundle_id

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        streamlines_data = cls(data['signal'].item(), data['gradients'].item())
        streamlines_data.filename = filename
        streamlines_data.streamlines._data = data['coords']
        streamlines_data.streamlines._offsets = data['offsets']
        streamlines_data.streamlines._lengths = data['lengths']
        streamlines_data.bundle_ids = data['bundle_ids']
        streamlines_data.name2id = OrderedDict([(str(k), int(v)) for k, v in data['name2id']])
        return streamlines_data

    def save(self, filename):
        np.savez(filename,
                 signal=self.signal,
                 gradients=self.gradients,
                 coords=self.streamlines._data.astype(np.float32),
                 offsets=self.streamlines._offsets,
                 lengths=self.streamlines._lengths.astype(np.int16),
                 bundle_ids=self.bundle_ids,
                 name2id=list(self.name2id.items()))

    def __str__(self):
        import textwrap
        msg = textwrap.dedent("""
                              ################################################
                              Dataset "{dataset_name}"
                              ################################################
                              ------------------- Streamlines ----------------
                              Nb. streamlines:    {nb_streamlines:}
                              Nb. bundles:        {nb_bundles}
                              Step sizes (in mm): {step_sizes}
                              Fiber nb. pts:      {fiber_lengths}
                              --------------------- Image --------------------
                              Dimension:     {dimension}
                              Voxel size:    {voxel_size}
                              Nb. B0 images: {nb_b0s}
                              Nb. gradients: {nb_gradients}
                              dwi filename:  {dwi_filename}
                              affine: {affine}
                              -------------------- Bundles -------------------
                              {bundles_infos}
                              """)

        name_max_length = max(map(len, self.name2id.keys()))
        bundles_infos = "\n".join([(name.ljust(name_max_length) +
                                    "{}".format((self.bundle_ids==bundle_id).sum()).rjust(12))
                                   for name, bundle_id in self.name2id.items()])

        t = nib.streamlines.Tractogram(self.streamlines.copy())
        t.apply_affine(self.signal.affine)  # Bring streamlines to RAS+mm
        step_sizes = np.sqrt(np.sum(np.diff(t.streamlines._data, axis=0)**2, axis=1))
        step_sizes = np.concatenate([step_sizes[o:o+l-1] for o, l in zip(t.streamlines._offsets, t.streamlines._lengths)])

        msg = msg.format(dataset_name=self.filename,
                         nb_streamlines=len(self.streamlines),
                         nb_bundles=len(self.name2id),
                         step_sizes="[{:.3f}, {:.3f}] (avg. {:.3f})".format(step_sizes.min(), step_sizes.max(), step_sizes.mean()),
                         fiber_lengths="[{}, {}] (avg. {:.1f})".format(self.streamlines._lengths.min(), self.streamlines._lengths.max(), self.streamlines._lengths.mean()),
                         dimension=self.signal.shape,
                         voxel_size=tuple(self.signal.header.get_zooms()),
                         nb_b0s=self.gradients.b0s_mask.sum(),
                         nb_gradients=np.logical_not(self.gradients.b0s_mask).sum(),
                         dwi_filename=self.signal.get_filename(),
                         affine="\n        ".join(str(self.signal.affine).split('\n')),
                         bundles_infos=bundles_infos)
        return msg[1:]  # Without the first newline.


class VolumeManager(object):
    def __init__(self):
        self.volumes = []
        self.volumes_strides = []

    @property
    def data_dimension(self):
        return self.volumes[0].get_value().shape[-1]

    def register(self, volume):
        volume_id = len(self.volumes)
        shape = np.array(volume.shape[:-1], dtype=floatX)
        strides = np.r_[1, np.cumprod(shape[::-1])[:-1]][::-1]
        self.volumes_strides.append(strides)
        self.volumes.append(sharedX(volume, name='volume_{}'.format(volume_id)))

        # Sanity check: make sure the size of the last dimension is the same for all volumes.
        assert self.data_dimension == volume.shape[-1]
        return volume_id

    def eval_at_coords(self, coords):
        data_at_coords = T.zeros((coords.shape[0], self.volumes[0].shape[-1]))
        for i, (volume, strides) in enumerate(zip(self.volumes, self.volumes_strides)):
            selection = T.eq(coords[:, 3], i).nonzero()[0]  # Theano's way of doing: coords[:, 3] == i
            selected_coords = coords[selection, :3]
            data_at_selected_coords = eval_volume_at_3d_coordinates_in_theano(volume, selected_coords, strides=strides)
            data_at_coords = T.set_subtensor(data_at_coords[selection], data_at_selected_coords)

        return data_at_coords


class MaskClassifierData(object):
    def __init__(self, signal, gradients, mask, positive_coords, negative_coords):
        """
        Parameters
        ----------
        signal: :class:`nibabel.Nifti1Image` object
            Diffusion signal used to generate the streamlines.
        gradients: :class:`dipy.core.gradients.GradientTable` object
            Diffusion gradient information for the `signal`.
        mask: :class:`nibabel.Nifti1Image` object
            3D binary mask image
        positive_coords: :class: `numpy.ndarray` object
            List of positive examples coordinates
        negative_coords: :class: `numpy.ndarray` object
            List of negative examples coordinates
        """
        self.signal = signal
        self.gradients = gradients
        self.mask = mask
        self.positive_coords = positive_coords
        self.negative_coords = negative_coords
        self.subject_id = None
        self.filename = None

    @property
    def volume(self):
        if self._volume is None:
            # Returns original signal
            return self.signal.get_data()

        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        mask_classifier_data = cls(data['signal'].item(), data['gradients'].item(), data['mask'].item(), data['positive_coords'],
                                   data['negative_coords'])
        mask_classifier_data.filename = filename
        return mask_classifier_data

    def save(self, filename):
        np.savez(filename,
                 signal=self.signal,
                 gradients=self.gradients,
                 mask=self.mask,
                 positive_coords=self.positive_coords,
                 negative_coords=self.negative_coords)

    def __str__(self):
        import textwrap
        msg = textwrap.dedent("""
                              ################################################
                              Dataset "{dataset_name}"
                              ################################################
                              --------------------- Image --------------------
                              Dimension:     {dimension}
                              Voxel size:    {voxel_size}
                              Nb. B0 images: {nb_b0s}
                              Nb. gradients: {nb_gradients}
                              dwi filename:  {dwi_filename}
                              affine: {affine}
                              ---------------------- Mask --------------------
                              Shape:         {mask_shape}
                              """)

        msg = msg.format(dataset_name=self.filename,
                         dimension=self.signal.shape,
                         voxel_size=tuple(self.signal.header.get_zooms()),
                         nb_b0s=self.gradients.b0s_mask.sum(),
                         nb_gradients=np.logical_not(self.gradients.b0s_mask).sum(),
                         dwi_filename=self.signal.get_filename(),
                         affine="\n        ".join(str(self.signal.affine).split('\n')),
                         mask_shape=self.mask.shape)
        return msg[1:]  # Without the first newline.


def map_coordinates_3d_4d(input_array, indices, affine=None, order=1):
    """ Evaluate the input_array data at the given indices
    using trilinear interpolation

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array

    Notes
    -----
    At some point this will be merged in Dipy. See PR #587.
    """
    if affine is not None:
        inv_affine = np.linalg.inv(affine)
        indices = (np.dot(indices, inv_affine[:3, :3]) + inv_affine[:3, 3])

    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=order)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=order)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def eval_volume_at_3d_coordinates(volume, coords):
    """ Evaluates the volume data at the given coordinates using trilinear interpolation.

    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.

    Returns
    -------
    output : 2D array
        Values from volume.
    """

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(volume, coords.T, order=1, mode="nearest")

    if volume.ndim == 4:
        values_4d = []
        for i in range(volume.shape[-1]):
            values_tmp = map_coordinates(volume[..., i],
                                         coords.T, order=1, mode="nearest")
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def normalize_dwi(weights, b0):
    """ Normalize dwi by the first b0.

    Parameters:
    -----------
    weights : ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    b0 : ndarray of shape (X, Y, Z)
        B0 image.

    Returns
    -------
    ndarray
        Diffusion weights normalized by the B0.
    """
    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure in every voxels weights are lower than ones from the b0.
    # Should not happen, but with the noise we never know!
    nb_erroneous_voxels = np.sum(weights > b0)
    if nb_erroneous_voxels != 0:
        print ("Nb. erroneous voxels: {}".format(nb_erroneous_voxels))
        weights = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[np.logical_not(np.isfinite(weights_normed))] = 0.

    return weights_normed


def get_spherical_harmonics_coefficients(dwi, bvals, bvecs, sh_order=8, smooth=0.006, first=False, mean_centering=True):
    """ Compute coefficients of the spherical harmonics basis.

    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
    bvals : ndarray shape (N,)
        B-values used with each direction.
    bvecs : ndarray shape (N, 3)
        Directions of the diffusion signal. Directions are
        assumed to be only on the hemisphere.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.
    mean_centering : bool
        If True, signal will have zero mean in each direction for all nonzero voxels

    Returns
    -------
    sh_coeffs : ndarray of shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number of
        coeffs depends on `sh_order`.
    """
    bvals = np.asarray(bvals)
    bvecs = np.asarray(bvecs)
    dwi_weights = dwi.get_data().astype("float32")

    # Exract the averaged b0.
    b0_idx = bvals == 0
    b0 = dwi_weights[..., b0_idx].mean(axis=3)

    # Extract diffusion weights and normalize by the b0.
    bvecs = bvecs[np.logical_not(b0_idx)]
    weights = dwi_weights[..., np.logical_not(b0_idx)]
    weights = normalize_dwi(weights, b0)

    # Assuming all directions are on the hemisphere.
    raw_sphere = HemiSphere(xyz=bvecs)

    # Fit SH to signal
    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, raw_sphere.theta, raw_sphere.phi)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L)
    data_sh = np.dot(weights, invB.T)

    if mean_centering:
        # Normalization in each direction (zero mean)
        idx = data_sh.sum(axis=-1).nonzero()
        means = data_sh[idx].mean(axis=0)
        data_sh[idx] -= means

    return data_sh


def resample_dwi(dwi, bvals, bvecs, directions=None, sh_order=8, smooth=0.006, mean_centering=True):
    """ Resamples a diffusion signal according to a set of directions using spherical harmonics.

    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
    bvals : ndarray shape (N,)
        B-values used with each direction.
    bvecs : ndarray shape (N, 3)
        Directions of the diffusion signal. Directions are
        assumed to be only on the hemisphere.
    directions : `dipy.core.sphere.Sphere` object, optional
        Directions the diffusion signal will be resampled to. Directions are
        assumed to be on the whole sphere, not the hemisphere like bvecs.
        If omitted, 100 directions evenly distributed on the sphere will be used.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.
    mean_centering : bool
        If True, signal will have zero mean in each direction for all nonzero voxels

    Returns
    -------
    ndarray
        Diffusion weights resampled according to `sphere`.
    """
    data_sh = get_spherical_harmonics_coefficients(dwi, bvals, bvecs, sh_order=sh_order, smooth=smooth, mean_centering=False)

    sphere = get_sphere('repulsion100')
    # sphere = get_sphere('repulsion724')
    if directions is not None:
        sphere = Sphere(xyz=bvecs[1:])

    sph_harm_basis = sph_harm_lookup.get('mrtrix')
    Ba, m, n = sph_harm_basis(sh_order, sphere.theta, sphere.phi)
    data_resampled = np.dot(data_sh, Ba.T)

    if mean_centering:
        # Normalization in each direction (zero mean)
        idx = data_resampled.sum(axis=-1).nonzero()
        means = data_resampled[idx].mean(axis=0)
        data_resampled[idx] -= means

    return data_resampled


def remove_similar_streamlines(streamlines, removal_distance=2.):
    """ Computes a distance matrix using all streamlines, then removes streamlines closer than `removal_distance`.

    Parameters
    -----------
    streamlines : `ArraySequence` object or list of 3D arrays
        Streamlines to downsample
    removal_distance : float
        Distance for which streamlines are considered 'similar' and should be removed

    Returns
    -------
    `ArraySequence` object
        Downsampled streamlines
    """
    # Simple trick to make it faster than using 40-60 points
    sample_10_streamlines = set_number_of_points(streamlines, 10)
    distance_matrix = distance_matrix_mdf(sample_10_streamlines, sample_10_streamlines)

    current_id = 0
    while True:
        indices = np.where(distance_matrix[current_id] < removal_distance)[0]

        it = 0
        if len(indices) > 1:
            for k in indices:
                # Every streamlines similar to yourself (excluding yourself)
                # should be deleted from the set of desired streamlines
                if not current_id == k:
                    streamlines.pop(k - it)
                    distance_matrix = np.delete(distance_matrix, k - it, axis=0)
                    distance_matrix = np.delete(distance_matrix, k - it, axis=1)
                    it += 1

        current_id += 1
        # Once you reach the end of the remaining streamlines
        if current_id >= len(streamlines):
            break

    return streamlines


def subsample_streamlines(streamlines, clustering_threshold=6., removal_distance=2.):
    """ Subsample a group of streamlines (should be used on streamlines from a single bundle or similar structure).
    Streamlines are first clustered using `clustering_threshold`, then for each cluster, similar streamlines (closer than `removal_distance`) are removed.

    Parameters
    ----------
    streamlines : `ArraySequence` object
        Streamlines to subsample
    clustering_threshold : float
        distance threshold for clustering (in the space of the tracks)
    removal_distance : float
        distance threshold for removal (in the space of the tracks)
    Returns
    -------
    `ArraySequence` object
        Downsampled streamlines
    """

    output_streamlines = []

    qb = QuickBundles(streamlines, dist_thr=clustering_threshold, pts=20)
    for i in range(len(qb.centroids)):
        temp_streamlines = qb.label2tracks(streamlines, i)
        output_streamlines.extend(remove_similar_streamlines(temp_streamlines, removal_distance=removal_distance))

    return output_streamlines
