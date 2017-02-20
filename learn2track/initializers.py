import numpy as np

import theano
from smartlearner.initializers import WeightInitializer


class OrthogonalInitializer(WeightInitializer):
    """ Orthogonal initialization of a weights matrix.

    This is a better (faster than qr approach) implementation than the one in smartlearner.
    This code was taken from https://gist.github.com/kastnerkyle/f7464d98fe8ca14f2a1a
    and was based/inspired from the one in Lasagne made by @benanne.
    """
    def _generate_array(self, dim):
        flat_shape = (dim[0], np.prod(dim[1:]))
        a = self.rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(dim)
        return q[:dim[0], :dim[1]].astype(theano.config.floatX)

class HeInitializer(WeightInitializer):
    """ He weight initialization. (For relu)

    References
    ----------
    [He2015] Kaiming He et al. (2015): Delving deep into rectifiers: Surpassing
             human-level performance on imagenet classification. arXiv:1502.01852.
    """

    def __call__(self, shared_var, gain=1.):
        dim = shared_var.get_value().shape
        shared_var.set_value(self._generate_array(dim, gain))
        return shared_var

    def _generate_array(self, dim, gain=1.):
        fan_in = dim[0]
        std = gain * np.sqrt(1. / fan_in)
        return np.asarray(self.rng.normal(loc=0, scale=std, size=dim), dtype=theano.config.floatX)
