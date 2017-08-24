import numpy as np
import theano
import theano.tensor as T
from smartlearner.interfaces import Loss
from theano.sandbox.rng_mrg import MRG_RandomStreams

from learn2track.models.gru_regression import GRU_Regression
from learn2track.models.layers import LayerRegression
from learn2track.utils import logsumexp, softmax, l2distance

floatX = theano.config.floatX


class GRU_Gaussian(GRU_Regression):
    """ A GRU_Regression model with the output size computed for a gaussian distribution, using a diagonal covariance matrix
    """

    def __init__(self, volume_manager, input_size, hidden_sizes, output_size, use_previous_direction=False, use_layer_normalization=False, drop_prob=0.,
                 use_zoneout=False, use_skip_connections=False, seed=1234, **_):
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
        use_previous_direction : bool
            Use the previous direction as an additional input
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
        super(GRU_Regression, self).__init__(input_size, hidden_sizes, use_layer_normalization=use_layer_normalization, drop_prob=drop_prob,
                                             use_zoneout=use_zoneout, use_skip_connections=use_skip_connections, seed=seed)
        self.volume_manager = volume_manager

        assert output_size == 3  # Only 3-dimensional target is supported for now
        self.output_size = output_size

        self.use_previous_direction = use_previous_direction

        # GRU_Gaussian does not predict a direction, so it cannot predict an offset
        self.predict_offset = False

        # Do not use dropout/zoneout in last hidden layer
        self.layer_regression_size = sum([output_size,  # Means
                                          output_size])  # Stds
        output_layer_input_size = sum(self.hidden_sizes) if self.use_skip_connections else self.hidden_sizes[-1]
        self.layer_regression = LayerRegression(output_layer_input_size, self.layer_regression_size)

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters['layer_regression_size'] = self.layer_regression_size
        hyperparameters['use_previous_direction'] = self.use_previous_direction
        return hyperparameters

    @staticmethod
    def get_distribution_parameters(regression_output):
        mu = regression_output[..., :3]
        sigma = T.exp(regression_output[..., 3:])
        return mu, sigma

    @staticmethod
    def _get_stochastic_samples(srng, mu, sigma):
        batch_size = mu.shape[0]

        # mu.shape : (batch_size, 3)
        # sigma.shape : (batch_size, 3)

        noise = srng.normal((batch_size, 3))
        samples = mu + sigma * noise

        return samples

    @staticmethod
    def _get_max_component_samples(mu, _):
        return mu

    def make_sequence_generator(self, subject_id=0, use_max_component=False):
        """ Makes functions that return the prediction for x_{t+1} for every
        sequence in the batch given x_{t} and the current state of the model h^{l}_{t}.

        Parameters
        ----------
        subject_id : int, optional
            ID of the subject from which its diffusion data will be used. Default: 0.
        use_max_component : bool, optional
            Use the maximum of the probability distribution instead of sampling values
        """

        # Build the sequence generator as a theano function.
        states_h = []
        for i in range(len(self.hidden_sizes)):
            state_h = T.matrix(name="layer{}_state_h".format(i))
            states_h.append(state_h)

        symb_x_t = T.matrix(name="x_t")

        new_states = self._fprop_step(symb_x_t, *states_h)
        new_states_h = new_states[:len(self.hidden_sizes)]

        # regression_output.shape : (batch_size, target_size)
        regression_output = new_states[-1]
        distribution_params = self.get_distribution_parameters(regression_output)

        if use_max_component:
            predictions = self._get_max_component_samples(*distribution_params)
        else:
            srng = MRG_RandomStreams(1234)
            predictions = self._get_stochastic_samples(srng, *distribution_params)

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
            next_states : list of 2D array of shape (batch_size, hidden_size)
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
            next_states = results[1:]
            return next_x_t, next_states

        return _gen


class GaussianNLL(Loss):
    """ Computes the negative log likelihood of a gaussian
    """

    def __init__(self, model, dataset, sum_over_timestep=False):
        super().__init__(model, dataset)
        self.d = model.output_size
        self.sum_over_timestep = sum_over_timestep

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, regression_layer_size)
        regression_outputs = model_output

        # mu.shape : (batch_size, seq_len, 3)
        # sigma.shape : (batch_size, seq_len, 3)
        mu, sigma = self.model.get_distribution_parameters(regression_outputs)

        # targets.shape : (batch_size, seq_len, 3)
        targets = self.dataset.symb_targets

        square_mahalanobis_dist = T.sum(T.square((targets - mu) / sigma), axis=-1)

        # loss_per_timestep.shape : (batch_size, seq_len)
        self.loss_per_time_step = 0.5 * (self.d * np.float32(np.log(2 * np.pi)) + 2 * T.sum(T.log(sigma), axis=-1) + square_mahalanobis_dist)

        # loss_per_seq.shape : (batch_size,)
        # loss_per_seq is the log probability for each sequence
        self.loss_per_seq = T.sum(self.loss_per_time_step * mask, axis=1)

        if not self.sum_over_timestep:
            # loss_per_seq is the average log probability for each sequence
            self.loss_per_seq /= T.sum(mask, axis=1)

        return self.loss_per_seq


class GaussianExpectedValueL2Distance(Loss):
    """ Computes the L2 distance between the target and the expected value of a gaussian distribution
    """

    def __init__(self, model, dataset):
        super().__init__(model, dataset)
        self.d = model.output_size

    def _get_updates(self):
        return {}  # There is no updates for L2Distance.

    def _compute_losses(self, model_output):
        mask = self.dataset.symb_mask

        # regression_outputs.shape = (batch_size, seq_length, regression_layer_size)
        regression_outputs = model_output

        # mu.shape : (batch_size, seq_len, 3)
        # sigma.shape : (batch_size, seq_len, 3)
        mu, sigma = self.model.get_distribution_parameters(regression_outputs)

        # targets.shape : (batch_size, seq_len, 3)
        targets = self.dataset.symb_targets

        # samples.shape : (batch_size, seq_len, 3)
        self.samples = mu

        # loss_per_time_step.shape = (batch_size, seq_len)
        self.loss_per_time_step = l2distance(self.samples, targets)
        # loss_per_seq.shape = (batch_size,)
        self.loss_per_seq = T.sum(self.loss_per_time_step * mask, axis=1) / T.sum(mask, axis=1)

        return self.loss_per_seq
