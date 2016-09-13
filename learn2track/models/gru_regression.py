import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T

from smartlearner.interfaces import Loss
import smartlearner.initializers as initer

from learn2track.models.layers import LayerRegression
from learn2track.models import GRU

floatX = theano.config.floatX


class GRU_Regression(GRU):
    """ A standard GRU model with a regression layer stacked on top of it.
    """
    def __init__(self, volume_manager, input_size, hidden_sizes, output_size, **_):
        """
        Parameters
        ----------
        volume_manager : :class:`VolumeManger` object
            Use to evaluate the diffusion signal at specific coordinates.
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
        output_size : int
            Number of units the regression layer should have.
        """
        super().__init__(input_size, hidden_sizes)
        self.volume_manager = volume_manager
        self.output_size = output_size
        self.layer_regression = LayerRegression(self.hidden_sizes[-1], self.output_size)

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters

    def _fprop_step(self, Xi, *args):
        # coords : streamlines 3D coordinates.
        # coords.shape : (batch_size, 4) where the last column is a dwi ID.
        # args.shape : n_layers * (batch_size, layer_size)
        coords = Xi

        # Get diffusion data.
        # data_at_coords.shape : (batch_size, input_size)
        data_at_coords = self.volume_manager.eval_at_coords(coords)

        # Hidden state to be passed to the next GRU iteration (next _fprop call)
        # next_hidden_state.shape : n_layers * (batch_size, layer_size)
        next_hidden_state = super()._fprop(data_at_coords, *args)

        # Compute the direction to follow for step (t)
        regression_out = self.layer_regression.fprop(next_hidden_state[-1])

        return next_hidden_state + (regression_out,)

    def get_output(self, X):
        # X.shape : (batch_size, seq_len, n_features=4)
        # For tractography n_features is (x,y,z) + (dwi_id,)

        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop_step,
                                       # We want to scan over sequence elements, not the examples.
                                       sequences=[T.transpose(X, axes=(1, 0, 2))],
                                       outputs_info=outputs_info_h + [None],
                                       non_sequences=self.parameters + self.volume_manager.volumes,
                                       strict=True,
                                       #truncate_gradient=100
                                       )

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        # regression_out.shape : (batch_size, seq_len, target_size=3)
        self.regression_out = T.transpose(results[-1], axes=(1, 0, 2))
        return self.regression_out

    def seq_next(self, x_t, subject_ids=None):
        """ Returns the prediction for x_{t+1} for every sequence in the batch.

        Parameters
        ----------
        x_t : ndarray with shape (batch_size, 3)
            Streamline coordinate (x, y, z).
        subject_ids : ndarray with shape (batch_size, 1), optional
            ID of the subject from which its diffusion data will be used. Default: [0]*len(x_t)
        """
        if subject_ids is None:
            subject_ids = np.array([0] * len(x_t), dtype=floatX)[:, None]

        # Append the DWI ID of each sequence after the 3D coordinates.
        x_t = np.c_[x_t, subject_ids]

        if self._gen is None:
            # Build theano function and cache it.
            self.seq_reset(batch_size=len(x_t))

            symb_x_t = T.TensorVariable(type=T.TensorType("floatX", [False] * x_t.ndim), name='x_t')
            symb_x_t.tag.test_value = x_t

            states = self.states_h + [0]
            new_states = self._fprop_step(symb_x_t, *states)
            new_states_h = new_states[:len(self.hidden_sizes)]

            # output.shape : (batch_size, target_size)
            next_step_predictions = new_states[-1]

            updates = OrderedDict()
            for i in range(len(self.hidden_sizes)):
                updates[self.states_h[i]] = new_states_h[i]

            self._gen = theano.function([symb_x_t], next_step_predictions, updates=updates)

        return self._gen(x_t)

    def use(self, X):
        directions = self.get_output(X)
        return directions


class L2DistanceForSequences(Loss):
    """ Computes the L2 error of the output.

    Notes
    -----
    This loss assumes the regression target at every time step is a vector.
    """
    def __init__(self, model, dataset, normalize_output=False, eps=1e-6):
        super().__init__(model, dataset)
        self.normalize_output = normalize_output
        self.eps = eps

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, out_dim)
        regression_outputs = model_output
        if self.normalize_output:
            regression_outputs /= (T.sqrt(T.sum(regression_outputs**2, axis=2, keepdims=True) + self.eps))

        self.regression_outputs = regression_outputs  # dbg
        # L2_errors_per_time_step.shape = (batch_size,)
        # self.L2_errors_per_time_step = T.sqrt(T.sum(((regression_outputs - self.dataset.symb_targets)**2), axis=2))  # NaN
        self.L2_errors_per_time_step = T.sum(((regression_outputs - self.dataset.symb_targets)**2), axis=2)
        # avg_L2_error_per_seq.shape = (batch_size,)
        self.avg_L2_error_per_seq = T.sum(self.L2_errors_per_time_step*mask, axis=1) / T.sum(mask, axis=1)

        return self.avg_L2_error_per_seq
