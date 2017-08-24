import numpy as np
import smartlearner.initializers as initer
import theano
import theano.tensor as T
from smartlearner.interfaces import Loss

from learn2track.models import GRU
from learn2track.models.layers import LayerDense
from learn2track.utils import l2distance

floatX = theano.config.floatX


class GRU_Regression(GRU):
    """ A standard GRU model with a regression layer stacked on top of it.
    """

    def __init__(self, volume_manager, input_size, hidden_sizes, output_size, activation='tanh', use_previous_direction=False, predict_offset=False,
                 use_layer_normalization=False, drop_prob=0., use_zoneout=False, use_skip_connections=False, seed=1234, **_):
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
        activation : str
            Activation function to apply on the "cell candidate"
        use_previous_direction : bool
            Use the previous direction as an additional input
        predict_offset : bool
            Predict the offset from the previous direction instead (need use_previous_direction).
        use_layer_normalization : bool
            Use LayerNormalization to normalize preactivations and stabilize hidden layer evolution
        drop_prob : float
            Dropout/Zoneout probability for recurrent networks. See: https://arxiv.org/pdf/1512.05287.pdf & https://arxiv.org/pdf/1606.01305.pdf
        use_zoneout : bool
            Use zoneout implementation instead of dropout
        use_skip_connections : bool
            Use skip connections from the input to all hidden layers in the network, and from all hidden layers to the output layer
        seed : int
            Random seed used for dropout normalization
        """
        super().__init__(input_size, hidden_sizes, activation=activation, use_layer_normalization=use_layer_normalization, drop_prob=drop_prob,
                         use_zoneout=use_zoneout, use_skip_connections=use_skip_connections, seed=seed)
        self.volume_manager = volume_manager
        self.output_size = output_size
        self.use_previous_direction = use_previous_direction
        self.predict_offset = predict_offset

        if self.predict_offset:
            assert self.use_previous_direction  # Need previous direction to predict offset.

        # Do not use dropout/zoneout in last hidden layer
        layer_regression_activation = "tanh" if self.predict_offset else "identity"
        output_layer_input_size = sum(self.hidden_sizes) if self.use_skip_connections else self.hidden_sizes[-1]
        self.layer_regression = LayerDense(output_layer_input_size, self.output_size, activation=layer_regression_activation, name="GRU_Regression")

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        super().initialize(weights_initializer)
        self.layer_regression.initialize(weights_initializer)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['output_size'] = self.output_size
        hyperparameters['use_previous_direction'] = self.use_previous_direction
        hyperparameters['predict_offset'] = self.predict_offset
        return hyperparameters

    @property
    def parameters(self):
        return super().parameters + self.layer_regression.parameters

    def _fprop_step(self, Xi, *args):
        # Xi.shape : (batch_size, 4)    *if self.use_previous_direction, Xi.shape : (batch_size,7)
        # coords + dwi ID (+ previous_direction)

        # coords : streamlines 3D coordinates.
        # coords.shape : (batch_size, 4) where the last column is a dwi ID.
        # args.shape : n_layers * (batch_size, layer_size)
        coords = Xi[:, :4]

        # Get diffusion data.
        # data_at_coords.shape : (batch_size, input_size)
        data_at_coords = self.volume_manager.eval_at_coords(coords)

        if self.use_previous_direction:
            # previous_direction.shape : (batch_size, 3)
            previous_direction = Xi[:, 4:]
            fprop_input = T.concatenate([data_at_coords, previous_direction], axis=1)
        else:
            fprop_input = data_at_coords

        # Hidden state to be passed to the next GRU iteration (next _fprop call)
        # next_hidden_state.shape : n_layers * (batch_size, layer_size)
        next_hidden_state = super()._fprop(fprop_input, *args)

        # Compute the direction to follow for step (t)

        output_layer_input = T.concatenate(next_hidden_state, axis=-1) if self.use_skip_connections else next_hidden_state[-1]
        regression_out = self.layer_regression.fprop(output_layer_input)

        if self.predict_offset:
            regression_out += previous_direction  # Skip-connection from the previous direction.

        return next_hidden_state + (regression_out,)

    def get_output(self, X):
        # X.shape : (batch_size, seq_len, n_features=[4|7])
        # For tractography n_features is (x,y,z) + (dwi_id,) + [previous_direction]

        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop_step,
                                       # We want to scan over sequence elements, not the examples.
                                       sequences=[T.transpose(X, axes=(1, 0, 2))],
                                       outputs_info=outputs_info_h + [None],
                                       non_sequences=self.parameters + self.volume_manager.volumes,
                                       strict=True)

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        # regression_out.shape : (batch_size, seq_len, target_size=3)
        self.regression_out = T.transpose(results[-1], axes=(1, 0, 2))
        return self.regression_out

    def make_sequence_generator(self, subject_id=0, **_):
        """ Makes functions that return the prediction for x_{t+1} for every
        sequence in the batch given x_{t} and the current state of the model h^{l}_{t}.

        Parameters
        ----------
        subject_id : int, optional
            ID of the subject from which its diffusion data will be used. Default: 0.
        """

        # Build the sequence generator as a theano function.
        states_h = []
        for i in range(len(self.hidden_sizes)):
            state_h = T.matrix(name="layer{}_state_h".format(i))
            states_h.append(state_h)

        symb_x_t = T.matrix(name="x_t")

        new_states = self._fprop_step(symb_x_t, *states_h)
        new_states_h = new_states[:len(self.hidden_sizes)]

        # predictions.shape : (batch_size, target_size)
        predictions = new_states[-1]

        f = theano.function(inputs=[symb_x_t] + states_h,
                            outputs=[predictions] + list(new_states_h))

        def _gen(x_t, states, previous_direction=None):
            """ Returns the prediction for x_{t+1} for every
                sequence in the batch given x_{t} and the current states
                of the model h^{l}_{t}.

            Parameters
            ----------
            x_t : ndarray with shape (batch_size, 3)
                Streamline coordinate (x, y, z).
            states : list of 2D array of shape (batch_size, hidden_size)
                Currrent states of the network.
            previous_direction : ndarray with shape (batch_size, 3)
                If using previous direction, these should be added to the input

            Returns
            -------
            next_x_t : ndarray with shape (batch_size, 3)
                Directions to follow.
            new_states : list of 2D array of shape (batch_size, hidden_size)
                Updated states of the network after seeing x_t.
            """
            # Append the DWI ID of each sequence after the 3D coordinates.
            subject_ids = np.array([subject_id] * len(x_t), dtype=floatX)[:, None]

            if not self.use_previous_direction:
                x_t = np.c_[x_t, subject_ids]
            else:
                x_t = np.c_[x_t, subject_ids, previous_direction]

            results = f(x_t, *states)
            next_x_t = results[0]
            new_states = results[1:]
            return next_x_t, new_states

        return _gen


class L2DistanceForSequences(Loss):
    """ Computes the L2 error of the output.

    Notes
    -----
    This loss assumes the regression target at every time step is a vector.
    """
    def __init__(self, model, dataset, normalize_output=False, eps=1e-6, sum_over_timestep=False):
        super().__init__(model, dataset)
        self.normalize_output = normalize_output
        self.eps = eps
        self.sum_over_timestep = sum_over_timestep

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, out_dim)
        regression_outputs = model_output
        if self.normalize_output:
            regression_outputs /= l2distance(regression_outputs, keepdims=True, eps=self.eps)

        self.samples = regression_outputs

        # loss_per_time_step.shape = (batch_size, seq_len)
        self.loss_per_time_step = l2distance(self.samples, self.dataset.symb_targets, eps=self.eps)
        # loss_per_seq.shape = (batch_size,)
        self.loss_per_seq = T.sum(self.loss_per_time_step * mask, axis=1)

        if not self.sum_over_timestep:
            self.loss_per_seq /= T.sum(mask, axis=1)

        return self.loss_per_seq
