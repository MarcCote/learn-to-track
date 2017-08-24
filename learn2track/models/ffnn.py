from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
import smartlearner.initializers as initer
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from learn2track.models.layers import LayerDense, LayerDenseNormalized
from smartlearner import utils as smartutils
from smartlearner.interfaces import Model

floatX = theano.config.floatX


class FFNN(Model):
    """ A standard FFNN model with no output layer.

    The output is simply the state of the last hidden layer.
    """

    def __init__(self, input_size, hidden_sizes, activation='tanh', use_layer_normalization=False, dropout_prob=0., use_skip_connections=False, seed=1234):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element X has.
        hidden_sizes : int, list of int
            Number of hidden units the model should have.
        activation : str
            Name of the activation function to use in the hidden layers
        use_layer_normalization : bool
            Use LayerNormalization to normalize preactivations
        dropout_prob : float
            Dropout probability for recurrent networks. See: https://arxiv.org/pdf/1512.05287.pdf
        use_skip_connections : bool
            Use skip connections from the input to all hidden layers in the network, and from all hidden layers to the output layer
        seed : int
            Random seed used for dropout normalization
        """
        self.graph_updates = OrderedDict()
        self._gen = None

        self.input_size = input_size
        self.hidden_sizes = [hidden_sizes] if type(hidden_sizes) is int else hidden_sizes
        self.activation = activation
        self.use_layer_normalization = use_layer_normalization
        self.dropout_prob = dropout_prob
        self.use_skip_connections = use_skip_connections
        self.seed = seed
        self.srng = MRG_RandomStreams(self.seed)

        layer_class = LayerDense
        if self.use_layer_normalization:
            layer_class = LayerDenseNormalized

        self.layers = []
        last_hidden_size = self.input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.layers.append(layer_class(last_hidden_size, hidden_size, activation=activation, name="Dense{}".format(i)))
            last_hidden_size = hidden_size

        self.dropout_vectors = {}
        if self.dropout_prob:
            p = 1 - self.dropout_prob
            for layer in self.layers:
                self.dropout_vectors[layer.name] = self.srng.binomial(size=(layer.input_size,), n=1, p=p, dtype=floatX) / p

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        for layer in self.layers:
            layer.initialize(weights_initializer)

    @property
    def updates(self):
        return self.graph_updates

    @property
    def hyperparameters(self):
        hyperparameters = {'version': 1,
                           'input_size': self.input_size,
                           'hidden_sizes': self.hidden_sizes,
                           'activation': self.activation,
                           'use_layer_normalization': self.use_layer_normalization,
                           'dropout_prob': self.dropout_prob,
                           'use_skip_connections': self.use_skip_connections,
                           'seed': self.seed}

        return hyperparameters

    @property
    def parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.parameters

        return parameters

    def get_init_states(self, batch_size):
        states_h = []
        for i, hidden_size in enumerate(self.hidden_sizes):
            state_h = np.zeros((batch_size, hidden_size), dtype=floatX)
            states_h.append(state_h)

        return states_h

    def _fprop(self, Xi):
        layers_h = []

        next_input = Xi
        for i, layer in enumerate(self.layers):
            dropout_W = self.dropout_vectors[layer.name] if self.dropout_prob else None
            layer_output = layer.fprop(next_input, dropout_W)
            layers_h.append(layer_output)
            if self.use_skip_connections:
                next_input = T.concatenate([layer_output, Xi], axis=-1)
            else:
                next_input = layer_output

        return tuple(layers_h)

    def get_output(self, X):
        last_layer_output = self._fprop(X)[-1]
        return last_layer_output

    def save(self, path):
        savedir = smartutils.create_folder(pjoin(path, type(self).__name__))
        smartutils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), self.hyperparameters)

        params = {param.name: param.get_value() for param in self.parameters}
        assert len(self.parameters) == len(params)  # Implies names are all unique.
        np.savez(pjoin(savedir, "params.npz"), **params)

        state = {
            "version": 1,
            "_srng_rstate": self.srng.rstate,
            "_srng_state_updates": [state_update[0].get_value() for state_update in self.srng.state_updates]}
        np.savez(pjoin(savedir, "state.npz"), **state)

    def load(self, path):
        loaddir = pjoin(path, type(self).__name__)

        parameters = np.load(pjoin(loaddir, "params.npz"))
        for param in self.parameters:
            param.set_value(parameters[param.name])

        state = np.load(pjoin(loaddir, 'state.npz'))
        self.srng.rstate[:] = state['_srng_rstate']
        for state_update, saved_state in zip(self.srng.state_updates, state["_srng_state_updates"]):
            state_update[0].set_value(saved_state)

    @classmethod
    def create(cls, path, **kwargs):
        loaddir = pjoin(path, cls.__name__)
        hyperparams = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))
        hyperparams.update(kwargs)

        model = cls(**hyperparams)
        model.load(path)
        return model
