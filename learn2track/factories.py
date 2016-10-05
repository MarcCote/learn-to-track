import theano.tensor as T

import smartlearner.initializers as initer
from learn2track.initializers import OrthogonalInitializer


WEIGHTS_INITIALIZERS = ["uniform", "zeros", "diagonal", "orthogonal", "gaussian"]


def weigths_initializer_factory(name, seed=1234):
    if name == "uniform":
        return initer.UniformInitializer(seed)
    elif name == "zeros":
        return initer.ZerosInitializer(seed)
    elif name == "diagonal":
        return initer.DiagonalInitializer(seed)
    elif name == "orthogonal":
        return OrthogonalInitializer(seed)
    elif name == "gaussian":
        return initer.GaussienInitializer(seed)

    raise NotImplementedError("Unknown: " + str(name))


ACTIVATION_FUNCTIONS = ["sigmoid", "hinge", "softplus", "tanh"]


def make_activation_function(name):
    if name == "sigmoid":
        return T.nnet.sigmoid
    elif name == "identity":
        return lambda x: x
    elif name == "hinge":
        return lambda x: T.maximum(x, 0.0)
    elif name == "softplus":
        return T.nnet.softplus
    elif name == "tanh":
        return T.tanh

    raise NotImplementedError("Unknown: " + str(name))


def optimizer_factory(hyperparams, loss):
    # Set learning rate method that will be used.
    if hyperparams["SGD"] is not None:
        from smartlearner.optimizers import SGD
        from smartlearner.direction_modifiers import ConstantLearningRate
        options = hyperparams["SGD"].split()
        optimizer = SGD(loss=loss)
        optimizer.append_direction_modifier(ConstantLearningRate(lr=float(options[0])))
        return optimizer

    elif hyperparams["AdaGrad"] is not None:
        from smartlearner.optimizers import AdaGrad
        options = hyperparams["AdaGrad"].split()
        lr = float(options[0])
        eps = float(options[1]) if len(options) > 1 else 1e-6
        return AdaGrad(loss=loss, lr=lr, eps=eps)

    elif hyperparams["Adam"] is not None:
        from smartlearner.optimizers import Adam
        options = hyperparams["Adam"].split()
        lr = float(options[0]) if len(options) > 0 else 0.0001
        return Adam(loss=loss, lr=lr)

    elif hyperparams["RMSProp"] is not None:
        from smartlearner.optimizers import RMSProp
        lr = float(hyperparams["RMSProp"])
        return RMSProp(loss=loss, lr=lr)

    elif hyperparams["Adadelta"]:
        from smartlearner.optimizers import Adadelta
        return Adadelta(loss=loss)

    else:
        raise ValueError("The optimizer is mandatory!")


def model_factory(hyperparams, input_size, output_size, volume_manager):
    if hyperparams['model'] == 'gru_regression' and hyperparams['learn_to_stop']:
        raise NotImplementedError()
        # from learn2track.models import GRU_RegressionAndBinaryClassification
        # return GRU_RegressionAndBinaryClassification(batch_scheduler.input_size,
        #                                              hyperparams['hidden_sizes'],
        #                                              batch_scheduler.target_size)

    elif hyperparams['model'] == 'gru_regression':
        from learn2track.models import GRU_Regression
        return GRU_Regression(volume_manager=volume_manager,
                              input_size=input_size,
                              hidden_sizes=hyperparams['hidden_sizes'],
                              output_size=output_size)

    elif hyperparams['model'] == 'gru_multistep':
        from learn2track.models import GRU_Multistep_Gaussian
        return GRU_Multistep_Gaussian(volume_manager=volume_manager,
                                      input_size=input_size,
                                      hidden_sizes=hyperparams['hidden_sizes'],
                                      target_dims=output_size,
                                      k=hyperparams['k'],
                                      m=hyperparams['m'],
                                      seed=hyperparams['seed'])

    else:
        raise ValueError("Unknown model!")


def loss_factory(hyperparams, model, dataset):
    if hyperparams['model'] == 'gru_regression' and hyperparams['learn_to_stop']:
        raise NotImplementedError()
        # from learn2track.models.gru_regression_and_binary_classification import L2DistancePlusBinaryCrossEntropy
        # return L2DistancePlusBinaryCrossEntropy(model, dataset, normalize_output=hyperparams["normalize"])

    elif hyperparams['model'] == 'gru_regression':
        from learn2track.models.gru_regression import L2DistanceForSequences
        return L2DistanceForSequences(model, dataset, normalize_output=hyperparams["normalize"])

    elif hyperparams['model'] == 'gru_multistep':
        from learn2track.models.gru_msp import MultistepMultivariateGaussianLossForSequences
        return MultistepMultivariateGaussianLossForSequences(model, dataset)

    else:
        raise ValueError("Unknown model!")


def batch_scheduler_factory(hyperparams, dataset, noisy_streamlines_sigma, shuffle_streamlines):
    if hyperparams['model'] == 'gru_regression':
        from learn2track.batch_schedulers import TractographyBatchScheduler
        return TractographyBatchScheduler(dataset,
                                          batch_size=hyperparams['batch_size'],
                                          noisy_streamlines_sigma=noisy_streamlines_sigma,
                                          seed=hyperparams['seed'],
                                          normalize_target=hyperparams['normalize'],
                                          shuffle_streamlines=shuffle_streamlines)

    elif hyperparams['model'] == 'gru_multistep':
        from learn2track.batch_schedulers import MultistepSequenceBatchScheduler
        return MultistepSequenceBatchScheduler(dataset,
                                               batch_size=hyperparams['batch_size'],
                                               k=hyperparams['k'],
                                               noisy_streamlines_sigma=noisy_streamlines_sigma,
                                               seed=hyperparams['seed'],
                                               shuffle_streamlines=shuffle_streamlines)
    else:
        raise ValueError("Unknown model!")