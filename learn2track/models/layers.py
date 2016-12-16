import numpy as np
import theano.tensor as T

from learn2track import factories

from smartlearner.utils import sharedX
import smartlearner.initializers as initer

from learn2track.utils import l2distance
from learn2track.initializers import OrthogonalInitializer



class LayerDense(object):
    def __init__(self, input_size, output_size, activation="identity", name="Dense"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)

        # Regression output weights and biases
        self.W = sharedX(value=np.zeros((self.input_size, self.output_size)), name=self.name+'_W')
        self.b = sharedX(value=np.zeros(output_size), name=self.name+'_b')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)

    @property
    def parameters(self):
        return [self.W, self.b]

    def fprop(self, X):
        preactivation = T.dot(X, self.W) + self.b
        out = self.activation_fct(preactivation)
        return out


class LayerRegression(object):
    def __init__(self, input_size, output_size, normed=False, name="Regression"):

        self.input_size = input_size
        self.output_size = output_size
        self.normed = normed
        self.name = name

        # Regression output weights and biases
        self.W = sharedX(value=np.zeros((self.input_size, self.output_size)), name=self.name+'_W')
        self.b = sharedX(value=np.zeros(output_size), name=self.name+'_b')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)

    @property
    def parameters(self):
        return [self.W, self.b]

    def fprop(self, X):
        out = T.dot(X, self.W) + self.b
        # Normalize the output vector.
        if self.normed:
            out /= l2distance(out, keepdims=True, eps=1e-8)

        return out


class LayerSoftmax(object):
    def __init__(self, input_size, output_size, name="Softmax"):

        self.input_size = input_size
        self.output_size = output_size
        self.name = name

        # Regression output weights and biases
        self.W = sharedX(value=np.zeros((self.input_size, self.output_size)), name=self.name+'_W')
        self.b = sharedX(value=np.zeros(output_size), name=self.name+'_b')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)

    @property
    def parameters(self):
        return [self.W, self.b]

    def fprop(self, X):
        preactivation = T.dot(X, self.W) + self.b
        # The softmax function, applied to a matrix, computes the softmax values row-wise.
        out = T.nnet.softmax(preactivation)
        return out


class LayerLstmWithPeepholes(object):
    def __init__(self, input_size, hidden_size, activation="tanh", name="LSTM"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)

        # Input weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the weights in that order: Wi, Wo, Wf, Wm
        self.W = sharedX(value=np.zeros((input_size, 4*hidden_size)), name=self.name+'_W')

        # Biases (i:input, o:output, f:forget, m:memory)
        # Concatenation of the biases in that order: bi, bo, bf, bm
        self.b = sharedX(value=np.zeros(4*hidden_size), name=self.name+'_b')

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the recurrence weights in that order: Ui, Uo, Uf, Um
        self.U = sharedX(value=np.zeros((hidden_size, 4*hidden_size)), name=self.name+'_U')

        # Peepholes (i:input, o:output, f:forget, m:memory)
        self.Vi = sharedX(value=np.ones(hidden_size), name=self.name+'_Vi')
        self.Vo = sharedX(value=np.ones(hidden_size), name=self.name+'_Vo')
        self.Vf = sharedX(value=np.ones(hidden_size), name=self.name+'_Vf')

    def initialize(self, weights_initializer=initer.UniformInitializer(1234)):
        weights_initializer(self.W)
        weights_initializer(self.U)

    @property
    def parameters(self):
        return [self.W, self.U, self.b,
                self.Vi, self.Vo, self.Vf]

    def fprop(self, Xi, last_h, last_m):
        def slice_(x, no):
            if type(no) is str:
                no = ['i', 'o', 'f', 'm'].index(no)
            return x[:, no*self.hidden_size: (no+1)*self.hidden_size]

        # SPEEDUP: compute the first linear transformation outside the scan i.e. for all timestep at once.
        # EDIT: I try and didn't see much speedup!
        Xi = (T.dot(Xi, self.W) + self.b)
        preactivation = Xi + T.dot(last_h, self.U)

        gate_i = T.nnet.sigmoid(slice_(preactivation, 'i') + last_m*self.Vi)
        mi = self.activation_fct(slice_(preactivation, 'm'))

        gate_f = T.nnet.sigmoid(slice_(preactivation, 'f') + last_m*self.Vf)
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(slice_(preactivation, 'o') + m*self.Vo)
        h = gate_o * self.activation_fct(m)

        return h, m


class LayerGRU(object):
    """ Gated Recurrent Unit

    Either use _[Cho14] (faster) or _[Chung14] implementation.

    References
    ----------
    .. [Cho14] Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, Yoshua Bengio
               "On the Properties of Neural Machine Translation: Encoder–Decoder Approaches",
               https://arxiv.org/pdf/1409.1259v2.pdf
    .. [Chung14] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio
                 "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling",
                 http://arxiv.org/pdf/1412.3555v1.pdf, 2014
    """
    def __init__(self, input_size, hidden_size, activation="tanh", name="GRU", fast_implementation=False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)
        self.fast_implementation = fast_implementation

        # Input weights (z:update, r:reset, h:hidden)
        # Concatenation of the weights in that order: Wz, Wr, Wh
        self.W = sharedX(value=np.zeros((input_size, 3*hidden_size)), name=self.name+'_W')

        # Biases (z:update, r:reset, h:hidden)
        # Concatenation of the biases in that order: bz, br, bh
        self.b = sharedX(value=np.zeros(3*hidden_size), name=self.name+'_b')
        # self.bh = sharedX(value=np.zeros(hidden_size), name=self.name+'_bh')

        # Recurrence weights (z:update, r:reset, h:hidden)
        if self.fast_implementation:
            # Concatenation of the recurrence weights in that order: Uz, Ur, Uh
            self.U = sharedX(value=np.zeros((hidden_size, 3*hidden_size)), name=self.name+'_U')
        else:
            # Concatenation of the recurrence weights in that order: Uz, Ur
            self.U = sharedX(value=np.zeros((hidden_size, 2*hidden_size)), name=self.name+'_U')
            self.Uh = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uh')

        # Only used for initialization.
        self.init_Uz = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uz_init')
        self.init_Ur = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Ur_init')
        self.init_Uh = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uh_init')

    @property
    def parameters(self):
        if self.fast_implementation:
            return [self.W, self.b, self.U]
        else:
            return [self.W, self.b, self.U, self.Uh]

    def initialize(self, weights_initializer=OrthogonalInitializer(1234)):
        weights_initializer(self.W)

        # Initialize recurrence matrices separately.
        weights_initializer(self.init_Uz)
        weights_initializer(self.init_Ur)

        if self.fast_implementation:
            weights_initializer(self.init_Uh)
            U = np.concatenate([self.init_Uz.get_value(), self.init_Ur.get_value(), self.init_Uh.get_value()], axis=1)
        else:
            weights_initializer(self.Uh)
            U = np.concatenate([self.init_Uz.get_value(), self.init_Ur.get_value()], axis=1)

        self.U.set_value(U)

    def _compute_slice(self, no):
        if type(no) is str:
            if no == 'zr':
                return slice(0, 2*self.hidden_size)

            no = ['z', 'r', 'h'].index(no)

        return slice(no*self.hidden_size, (no+1)*self.hidden_size)

    def _slice(self, x, no):
        return x[:, self._compute_slice(no)]

    def project_X(self, X):
        return T.dot(X, self.W) + self.b

    def fprop(self, last_h, X=None, proj_X=None):
        """ Compute the fprop of a Gated Recurrent Unit.

        If proj_X is provided, X will be ignored. Specifically, proj_X
        represents: T.dot(X, W_in) + b_hid.

        Parameters
        ----------
        last_h : Last hidden state (batch_size, hidden_size)
        X : Input (batch_size, input_size)
        proj_X : Projection of the input plus bias (batch_size, hidden_size)
        """
        if proj_X is None:
            proj_X = self.project_X(X)

        if self.fast_implementation:
            proj_h = T.dot(last_h, self.U)
            preactivation = self._slice(proj_X, 'zr') + self._slice(proj_h, 'zr')
        else:
            preactivation = self._slice(proj_X, 'zr') + T.dot(last_h, self.U)

        gate_z = T.nnet.sigmoid(self._slice(preactivation, 'z'))  # Update gate
        gate_r = T.nnet.sigmoid(self._slice(preactivation, 'r'))  # Reset gate

        # Candidate activation
        if self.fast_implementation:
            c = self.activation_fct(self._slice(proj_X, 'h') + gate_r * self._slice(proj_h, 'h'))
        else:
            c = self.activation_fct(self._slice(proj_X, 'h') + T.dot(last_h*gate_r, self.Uh))

        h = (1-gate_z)*last_h + gate_z*c

        return h


class LayerLSTM(object):
    def __init__(self, input_size, hidden_size, activation, name="LSTM"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        self.activation = activation
        self.activation_fct = factories.make_activation_function(self.activation)

        # Recurrence weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the recurrence weights in that order: Ui, Uo, Uf, Um
        self.U = sharedX(value=np.zeros((hidden_size, 4*hidden_size)), name=self.name+'_U')

        # Input weights (i:input, o:output, f:forget, m:memory)
        # Concatenation of the weights in that order: Wi, Wo, Wf, Wm
        self.W = sharedX(value=np.zeros((input_size, 4*hidden_size)), name=self.name+'_W')

        # Biases (i:input, o:output, f:forget, m:memory)
        # Concatenation of the biases in that order: bi, bo, bf, bm
        self.b = sharedX(value=np.zeros(4*hidden_size), name=self.name+'_b')

        # Only used for initialization.
        self.init_Ui = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Ui_init')
        self.init_Uo = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uo_init')
        self.init_Uf = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Uf_init')
        self.init_Um = sharedX(value=np.zeros((hidden_size, hidden_size)), name=self.name+'_Um_init')

    @property
    def parameters(self):
        return [self.W, self.b, self.U]

    def initialize(self, weights_initializer=OrthogonalInitializer(1234)):
        weights_initializer(self.W)

        # Initialize recurrence matrices separately.
        weights_initializer(self.init_Ui)
        weights_initializer(self.init_Uo)
        weights_initializer(self.init_Uf)
        weights_initializer(self.init_Um)
        U = np.concatenate([self.init_Ui.get_value(), self.init_Uo.get_value(), self.init_Uf.get_value(), self.init_Um.get_value()], axis=1)
        self.U.set_value(U)

    def _compute_slice(self, no):
        if type(no) is str:
            no = ['i', 'o', 'f', 'm'].index(no)

        return slice(no*self.hidden_size, (no+1)*self.hidden_size)

    def _slice(self, x, no):
        return x[:, self._compute_slice(no)]

    def fprop(self, last_h, last_m, X=None, proj_X=None):
        """ Compute the fprop of a LSTM unit.

        If proj_X is provided, X will be ignored. Specifically, proj_X
        represents: T.dot(X, W_in) + b_hid.

        Parameters
        ----------
        last_h : Last hidden state (batch_size, hidden_size)
        last_m : Last memory cell (batch_size, hidden_size)
        X : Input (batch_size, input_size)
        proj_X : Projection of the input plus bias (batch_size, hidden_size)
        """
        if proj_X is None:
            proj_X = self.project_X(X)

        preactivation = proj_X + T.dot(last_h, self.U)

        gate_i = T.nnet.sigmoid(self._slice(preactivation, 'i'))
        mi = self.activation_fct(self._slice(preactivation, 'm'))

        gate_f = T.nnet.sigmoid(self._slice(preactivation, 'f'))
        m = gate_i*mi + gate_f*last_m

        gate_o = T.nnet.sigmoid(self._slice(preactivation, 'o'))
        h = gate_o * self.activation_fct(m)

        return h, m

    def project_X(self, X):
        return T.dot(X, self.W) + self.b
