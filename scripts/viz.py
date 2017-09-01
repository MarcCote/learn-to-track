#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
from collections import OrderedDict

import nibabel as nib

from dipy.viz import actor, window, widget
from dipy.viz import fvtk
from dipy.tracking.streamline import set_number_of_points


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


def horizon(tractograms, data, affine):

    rng = np.random.RandomState(42)
    slicer_opacity = .8

    ren = window.Renderer()
    global centroid_actors
    centroid_actors = []
    for streamlines in tractograms:

        print(' Number of streamlines loaded {} \n'.format(len(streamlines)))
        colors = rng.rand(3)
        ren.add(actor.line(streamlines, colors, opacity=1., lod_points=10 ** 5))

    class SimpleTrackBallNoBB(window.vtk.vtkInteractorStyleTrackballCamera):
        def HighlightProp(self, p):
            pass

    style = SimpleTrackBallNoBB()
    # very hackish way
    style.SetPickColor(0, 0, 0)
    # style.HighlightProp(None)
    show_m = window.ShowManager(ren, size=(1200, 900), interactor_style=style)
    show_m.initialize()

    if data is not None:
        #from dipy.core.geometry import rodrigues_axis_rotation
        #affine[:3, :3] = np.dot(affine[:3, :3], rodrigues_axis_rotation((0, 0, 1), 45))

        image_actor = actor.slicer(data, affine)
        image_actor.opacity(slicer_opacity)
        image_actor.SetInterpolate(False)
        ren.add(image_actor)

        ren.add(fvtk.axes((10, 10, 10)))

        last_value = [10]
        def change_slice(obj, event):
            new_value = int(np.round(obj.get_value()))
            if new_value == image_actor.shape[1] - 1 or new_value == 0:
                new_value = last_value[0] + np.sign(new_value - last_value[0])

            image_actor.display(None, new_value, None)
            obj.set_value(new_value)
            last_value[0] = new_value

        slider = widget.slider(show_m.iren, show_m.ren,
                               callback=change_slice,
                               min_value=0,
                               max_value=image_actor.shape[1] - 1,
                               value=image_actor.shape[1] / 2,
                               label="Move slice",
                               right_normalized_pos=(.98, 0.6),
                               size=(120, 0), label_format="%0.lf",
                               color=(1., 1., 1.),
                               selected_color=(0.86, 0.33, 1.))

        slider.SetAnimationModeToJump()

    global size
    size = ren.GetSize()
    # ren.background((1, 0.5, 0))
    ren.background((0, 0, 0))
    global picked_actors
    picked_actors = {}

    def win_callback(obj, event):
        global size
        if size != obj.GetSize():

            if data is not None:
                slider.place(ren)
            size = obj.GetSize()

    global centroid_visibility
    centroid_visibility = True

    show_m.initialize()
    show_m.add_window_callback(win_callback)
    show_m.render()
    show_m.start()


def add_noise_to_streamlines(streamlines, sigma, rng=np.random.RandomState(42)):
    # Add gaussian noise, N(0, self.sigma).
    noisy_streamlines = streamlines.copy()
    shape = noisy_streamlines._data.shape
    noisy_streamlines._data += sigma * rng.randn(*shape)
    return noisy_streamlines


def horizon_flow(input_files, noisy_streamlines_sigma=0., verbose=True):
    """ Horizon

    Parameters
    ----------
    input_files : variable string
    cluster : bool, optional
    cluster_thr : float, optional
    random_colors : bool, optional
    verbose : bool, optional
    length_lt : float, optional
    length_gt : float, optional
    clusters_lt : int, optional
    clusters_gt : int, optional
    noisy_streamlines_sigma : float, optional
    """

    filenames = input_files
    # glob(input_files)
    tractograms = []

    data = None
    affine = None
    for i, f in enumerate(filenames):
        if verbose:
            print('Loading file ...')
            print(f)
            print('\n')

        if f.endswith('.trk') or f.endswith('.tck'):
            streamlines = nib.streamlines.load(f).streamlines
            idx = np.arange(len(streamlines))
            rng = np.random.RandomState(42)
            rng.shuffle(idx)
            streamlines = streamlines[idx[:100]]

            if noisy_streamlines_sigma > 0. and i > 0:
                streamlines = add_noise_to_streamlines(streamlines, noisy_streamlines_sigma)

            tractograms.append(streamlines)

        if f.endswith('.npz'):
            tractography_data = TractographyData.load(f)
            # idx = np.arange(len(tractography_data.streamlines))
            # rng = np.random.RandomState(42)
            # rng.shuffle(idx)
            # tractography_data.streamlines = tractography_data.streamlines[idx[:200]]
            # tractograms.append(tractography_data.streamlines)


            # Take M streamlines per bundle, but increase the value if there is only 1 bundle (i.e. whole brain)
            bundle_names = sorted(tractography_data.name2id.keys())
            M = 200 if len(bundle_names) > 1 else 10000
            for k in bundle_names:
                bundle_id = tractography_data.name2id[k]
                bundle_streamlines = tractography_data.streamlines[tractography_data.bundle_ids == bundle_id]
                indices = np.random.choice(len(bundle_streamlines), M)
                streamlines = bundle_streamlines[indices].copy()
                streamlines._lengths = streamlines._lengths.astype("int64")
                streamlines = set_number_of_points(streamlines, nb_points=40)
                tractograms.append(streamlines)

            if hasattr(tractography_data, 'signal'):
                signal = tractography_data.signal.get_data()
                data = signal[:, :, :, 0]
                affine = np.eye(4)

        if f.endswith('.nii.gz') or f.endswith('.nii'):

            img = nib.load(f)
            data = img.get_data()
            affine = img.get_affine()
            if verbose:
                print(affine)

    # tmp save
    # tractogram = nib.streamlines.Tractogram(tractograms[0])
    # tractogram.apply_affine(img.affine)
    # nib.streamlines.save(tractogram, "tmp.tck")
    # exit()

    horizon(tractograms, data, affine)


if __name__ == '__main__':
    horizon_flow(input_files=sys.argv[1:])
