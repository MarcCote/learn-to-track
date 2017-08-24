import numpy as np
import smartlearner.initializers as initer
import theano
import theano.tensor as T
from collections import OrderedDict
from os.path import join as pjoin
from smartlearner import utils as smartutils
from smartlearner.interfaces import Model
from theano.sandbox.rng_mrg import MRG_RandomStreams

from learn2track.models.layers import LayerGRU, LayerGruNormalized

floatX = theano.config.floatX

class GRU(Model):
    """ A standard GRU model with no output layer.

    See GRU_softmax or GRU_regression for implementations with an output layer.

    The output is simply the state of the last hidden layer.
    """

    def __init__(self, input_size, hidden_sizes, activation='tanh', use_layer_normalization=False, drop_prob=0., use_zoneout=False, use_skip_connections=False,
                 seed=1234):
        """
        Parameters
        ----------
        input_size : int
            Number of units each element Xi in the input sequence X has.
        hidden_sizes : int, list of int
            Number of hidden units each GRU should have.
        activation : str
            Activation function to apply on the "cell candidate"
        use_layer_normalization : bool
            Use LayerNormalization to normalize preactivations and stabilize hidden layer evolution
        drop_prob : float
            Dropout/Zoneout probability for recurrent networks. See: https://arxiv.org/pdf/1512.05287.pdf & https://arxiv.org/pdf/1606.01305.pdf
        use_zoneout : bool
            Use zoneout implementation instead of dropout (a different zoneout mask will be use at each timestep)
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
        self.drop_prob = drop_prob
        self.use_zoneout = use_zoneout
        self.use_skip_connections = use_skip_connections
        self.seed = seed
        self.srng = MRG_RandomStreams(self.seed)

        layer_class = LayerGRU
        if self.use_layer_normalization:
            layer_class = LayerGruNormalized

        self.layers = []
        last_hidden_size = self.input_size
        for i, hidden_size in enumerate(self.hidden_sizes):
            self.layers.append(layer_class(last_hidden_size, hidden_size, activation=activation, name="GRU{}".format(i)))
            last_hidden_size = hidden_size + (input_size if self.use_skip_connections else 0)

        self.dropout_vectors = {}
        if self.drop_prob and not self.use_zoneout:
            p = 1 - self.drop_prob
            for layer in self.layers:
                self.dropout_vectors[layer.name] = self.srng.binomial(size=(layer.hidden_size,), n=1, p=p, dtype=floatX) / p

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        for layer in self.layers:
            layer.initialize(weights_initializer)

    @property
    def updates(self):
        return self.graph_updates

    @property
    def hyperparameters(self):
        hyperparameters = {'version': 2,
                           'input_size': self.input_size,
                           'hidden_sizes': self.hidden_sizes,
                           'activation': self.activation,
                           'use_layer_normalization': self.use_layer_normalization,
                           'drop_prob': self.drop_prob,
                           'use_zoneout': self.use_zoneout,
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

    def _fprop(self, Xi, *args):
        layers_h = []

        input = Xi
        for i, layer in enumerate(self.layers):
            drop_states = None
            drop_value = None
            if self.drop_prob:
                if self.use_zoneout:
                    drop_value = 1.
                    drop_states = self.srng.binomial((layer.hidden_size,), n=1, p=1 - self.drop_prob, dtype=floatX)
                else:
                    drop_value = 0.
                    drop_states = self.dropout_vectors[layer.name]

            last_h = args[i]
            h = layer.fprop(input, last_h, drop_states, drop_value)
            layers_h.append(h)
            if self.use_skip_connections:
                input = T.concatenate([h, Xi], axis=-1)
            else:
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

        if hyperparams['version'] < 2:
            hyperparams['drop_prob'] = hyperparams['dropout_prob']
            del hyperparams['dropout_prob']

        model = cls(**hyperparams)
        model.load(path)
        return model
