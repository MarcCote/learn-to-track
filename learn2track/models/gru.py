from os.path import join as pjoin

import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

from smartlearner.interfaces import Model
from smartlearner import utils as smartutils
from smartlearner.utils import sharedX
import smartlearner.initializers as initer

from learn2track.models.layers import LayerGRU, LayerRegression, LayerDense

floatX = theano.config.floatX

class GRU(Model):
    """ A standard GRU model with no output layer.

    See GRU_softmax or GRU_regression for implementations with an output layer.

    The output is simply the state of the last hidden layer.
    """
    def __init__(self, input_size, hidden_sizes):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each LSTM should have.
        """
        self.graph_updates = OrderedDict()
        self._gen = None

        self.input_size = input_size
        self.hidden_sizes = [hidden_sizes] if type(hidden_sizes) is int else hidden_sizes

        self.layers = []
        last_hidden_size = self.input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.layers.append(LayerGRU(last_hidden_size, hidden_size, name="GRU{}".format(i)))
            last_hidden_size = hidden_size

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
                           'hidden_sizes': self.hidden_sizes}

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

    def _fprop(self, Xi, *args):
        layers_h = []

        input = Xi
        for i, layer in enumerate(self.layers):
            last_h = args[i]
            h = layer.fprop(X=input, last_h=last_h)
            layers_h.append(h)
            input = h

        return tuple(layers_h)

    def get_output(self, X):
        outputs_info_h = []
        for hidden_size in self.hidden_sizes:
            outputs_info_h.append(T.zeros((X.shape[0], hidden_size)))

        results, updates = theano.scan(fn=self._fprop,
                                       outputs_info=outputs_info_h,
                                       sequences=[T.transpose(X, axes=(1, 0, 2))])  # We want to scan over sequence elements, not the examples.

        self.graph_updates = updates
        # Put back the examples so they are in the first dimension.
        self.h = T.transpose(results[0], axes=(1, 0, 2))
        return self.h

    def save(self, path):
        savedir = smartutils.create_folder(pjoin(path, type(self).__name__))
        smartutils.save_dict_to_json_file(pjoin(savedir, "hyperparams.json"), self.hyperparameters)

        params = {param.name: param.get_value() for param in self.parameters}
        assert len(self.parameters) == len(params)  # Implies names are all unique.
        np.savez(pjoin(savedir, "params.npz"), **params)

    def load(self, path):
        loaddir = pjoin(path, type(self).__name__)

        parameters = np.load(pjoin(loaddir, "params.npz"))
        for param in self.parameters:
            param.set_value(parameters[param.name])

    @classmethod
    def create(cls, path, **kwargs):
        loaddir = pjoin(path, cls.__name__)
        hyperparams = smartutils.load_dict_from_json_file(pjoin(loaddir, "hyperparams.json"))
        hyperparams.update(kwargs)

        model = cls(**hyperparams)
        model.load(path)
        return model
