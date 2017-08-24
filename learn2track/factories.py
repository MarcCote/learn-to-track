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


ACTIVATION_FUNCTIONS = ["sigmoid", "hinge", "softplus", "tanh", "selu"]


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
    elif name == "selu":
        def selu(x):
            # See "Self-normalizing Neural Networks": https://arxiv.org/abs/1706.02515
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            # Original implementation
            # return scale * T.where(x >= 0.0, x, alpha * (T.exp(x) - 1))

            # Alternative implementation, without T.where
            x_pos = (T.abs_(x) + x) / 2
            x_neg = x - x_pos
            x_neg = alpha * T.exp(x_neg) - alpha
            return scale * (x_neg + x_pos)

        return selu

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
                              output_size=output_size,
                              activation=hyperparams['activation'],
                              use_previous_direction=hyperparams['feed_previous_direction'],
                              predict_offset=hyperparams['predict_offset'],
                              use_layer_normalization=hyperparams['use_layer_normalization'],
                              drop_prob=hyperparams['drop_prob'],
                              use_zoneout=hyperparams['use_zoneout'],
                              use_skip_connections=hyperparams['skip_connections'],
                              seed=hyperparams['seed'])

    elif hyperparams['model'] == 'gru_multistep':
        from learn2track.models import GRU_Multistep_Gaussian
        return GRU_Multistep_Gaussian(volume_manager=volume_manager,
                                      input_size=input_size,
                                      hidden_sizes=hyperparams['hidden_sizes'],
                                      target_dims=output_size,
                                      k=hyperparams['k'],
                                      m=hyperparams['m'],
                                      seed=hyperparams['seed'],
                                      use_previous_direction=hyperparams['feed_previous_direction'],
                                      use_layer_normalization=hyperparams['use_layer_normalization'],
                                      drop_prob=hyperparams['drop_prob'],
                                      use_zoneout=hyperparams['use_zoneout'])

    elif hyperparams['model'] == 'gru_mixture':
        from learn2track.models import GRU_Mixture
        return GRU_Mixture(volume_manager=volume_manager,
                           input_size=input_size,
                           hidden_sizes=hyperparams['hidden_sizes'],
                           output_size=output_size,
                           n_gaussians=hyperparams['n_gaussians'],
                           activation=hyperparams['activation'],
                           use_previous_direction=hyperparams['feed_previous_direction'],
                           use_layer_normalization=hyperparams['use_layer_normalization'],
                           drop_prob=hyperparams['drop_prob'],
                           use_zoneout=hyperparams['use_zoneout'],
                           use_skip_connections=hyperparams['skip_connections'],
                           seed=hyperparams['seed'])

    elif hyperparams['model'] == 'gru_gaussian':
        from learn2track.models import GRU_Gaussian
        return GRU_Gaussian(volume_manager=volume_manager,
                            input_size=input_size,
                            hidden_sizes=hyperparams['hidden_sizes'],
                            output_size=output_size,
                            use_previous_direction=hyperparams['feed_previous_direction'],
                            use_layer_normalization=hyperparams['use_layer_normalization'],
                            drop_prob=hyperparams['drop_prob'],
                            use_zoneout=hyperparams['use_zoneout'],
                            use_skip_connections=hyperparams['skip_connections'],
                            seed=hyperparams['seed'])

    elif hyperparams['model'] == 'ffnn_regression':
        from learn2track.models import FFNN_Regression
        return FFNN_Regression(volume_manager=volume_manager,
                               input_size=input_size,
                               hidden_sizes=hyperparams['hidden_sizes'],
                               output_size=output_size,
                               activation=hyperparams['activation'],
                               use_previous_direction=hyperparams['feed_previous_direction'],
                               predict_offset=hyperparams['predict_offset'],
                               use_layer_normalization=hyperparams['use_layer_normalization'],
                               dropout_prob=hyperparams['dropout_prob'],
                               use_skip_connections=hyperparams['skip_connections'],
                               seed=hyperparams['seed'])

    else:
        raise ValueError("Unknown model!")


def loss_factory(hyperparams, model, dataset, loss_type=None):
    if hyperparams['model'] == 'gru_regression' and hyperparams['learn_to_stop']:
        raise NotImplementedError()
        # from learn2track.models.gru_regression_and_binary_classification import L2DistancePlusBinaryCrossEntropy
        # return L2DistancePlusBinaryCrossEntropy(model, dataset, normalize_output=hyperparams["normalize"])

    elif hyperparams['model'] == 'gru_regression':
        if loss_type == "l2_mean" or loss_type is None:
            from learn2track.models.gru_regression import L2DistanceForSequences
            return L2DistanceForSequences(model, dataset)
        elif loss_type == "l2_sum":
            from learn2track.models.gru_regression import L2DistanceForSequences
            return L2DistanceForSequences(model, dataset, sum_over_timestep=True)
        else:
            raise ValueError("loss_type not available for gru_regression: {}".format(loss_type))

    elif hyperparams['model'] == 'gru_multistep':
        if loss_type == 'expected_value' or loss_type == 'maximum_component':
            from learn2track.models.gru_msp import MultistepMultivariateGaussianExpectedValueL2Distance
            return MultistepMultivariateGaussianExpectedValueL2Distance(model, dataset)
        elif loss_type is None:
            from learn2track.models.gru_msp import MultistepMultivariateGaussianNLL
            return MultistepMultivariateGaussianNLL(model, dataset)
        else:
            raise ValueError("Unrecognized loss_type: {}".format(loss_type))

    elif hyperparams['model'] == 'gru_gaussian':
        if loss_type == 'expected_value' or loss_type == 'maximum_component':
            from learn2track.models.gru_gaussian import GaussianExpectedValueL2Distance
            return GaussianExpectedValueL2Distance(model, dataset)
        elif loss_type == "nll_sum":
            from learn2track.models.gru_gaussian import GaussianNLL
            return GaussianNLL(model, dataset, sum_over_timestep=True)
        elif loss_type is None:
            from learn2track.models.gru_gaussian import GaussianNLL
            return GaussianNLL(model, dataset)
        else:
            raise ValueError("Unrecognized loss_type: {}".format(loss_type))

    elif hyperparams['model'] == 'gru_mixture':
        if loss_type == 'expected_value':
            from learn2track.models.gru_mixture import MultivariateGaussianMixtureExpectedValueL2Distance
            return MultivariateGaussianMixtureExpectedValueL2Distance(model, dataset)
        elif loss_type == 'maximum_component':
            from learn2track.models.gru_mixture import MultivariateGaussianMixtureMaxComponentL2Distance
            return MultivariateGaussianMixtureMaxComponentL2Distance(model, dataset)
        elif loss_type is None or loss_type == "nll_mean":
            from learn2track.models.gru_mixture import MultivariateGaussianMixtureNLL
            return MultivariateGaussianMixtureNLL(model, dataset)
        elif loss_type == "nll_sum":
            from learn2track.models.gru_mixture import MultivariateGaussianMixtureNLL
            return MultivariateGaussianMixtureNLL(model, dataset, sum_over_timestep=True)
        else:
            raise ValueError("Unrecognized loss_type: {}".format(loss_type))

    elif hyperparams['model'] == 'ffnn_regression':
        if loss_type == 'expected_value':
            from learn2track.models.ffnn_regression import UndirectedL2Distance
            return UndirectedL2Distance(model, dataset, hyperparams['normalize'])
        else:
            from learn2track.models.ffnn_regression import CosineSquaredLoss
            return CosineSquaredLoss(model, dataset, normalize_output=hyperparams['normalize'])

    else:
        raise ValueError("Unknown model!")


def batch_scheduler_factory(hyperparams, dataset, train_mode=True, batch_size_override=None, use_data_augment=True):
    """
    Build the right batch scheduler for the model and chosen mode

    Parameters
    ----------
    hyperparams : dict
        model's training hyperparams
    dataset : :class:`TractographyDataset`
        Dataset from which to get the examples.
    train_mode : bool
        overrides certain hyperparameters if training the model or not
    batch_size_override : int
        override batch_size hyperparam
    use_data_augment : bool
        Feed streamlines in both directions (doubles the batch size)
    """
    batch_size = hyperparams['batch_size'] if batch_size_override is None else batch_size_override

    if hyperparams['model'] in ['gru_regression', 'gru_mixture', 'gru_gaussian']:
        from learn2track.batch_schedulers import TractographyBatchScheduler
        return TractographyBatchScheduler(dataset,
                                          batch_size=batch_size,
                                          use_data_augment=use_data_augment,
                                          seed=hyperparams['seed'],
                                          normalize_target=hyperparams['normalize'],
                                          noisy_streamlines_sigma=hyperparams['noisy_streamlines_sigma'],
                                          shuffle_streamlines=train_mode,
                                          resample_streamlines=(not hyperparams['keep_step_size']) and train_mode,
                                          feed_previous_direction=hyperparams['feed_previous_direction'],
                                          sort_streamlines_by_length=hyperparams['sort_streamlines'] and train_mode)

    elif hyperparams['model'] == 'gru_multistep':
        from learn2track.batch_schedulers import MultistepSequenceBatchScheduler
        return MultistepSequenceBatchScheduler(dataset,
                                               batch_size=batch_size,
                                               use_data_augment=use_data_augment,
                                               k=hyperparams['k'],
                                               seed=hyperparams['seed'],
                                               normalize_target=hyperparams['normalize'],
                                               noisy_streamlines_sigma=hyperparams['noisy_streamlines_sigma'],
                                               shuffle_streamlines=train_mode,
                                               resample_streamlines=(not hyperparams['keep_step_size']) and train_mode,
                                               feed_previous_direction=hyperparams['feed_previous_direction'])
    elif hyperparams['model'] == 'ffnn_regression':
        from learn2track.batch_schedulers import SingleInputTractographyBatchScheduler
        return SingleInputTractographyBatchScheduler(dataset,
                                                     batch_size=batch_size,
                                                     use_data_augment=use_data_augment,
                                                     seed=hyperparams['seed'],
                                                     normalize_target=hyperparams['normalize'],
                                                     noisy_streamlines_sigma=hyperparams['noisy_streamlines_sigma'],
                                                     shuffle_streamlines=train_mode,
                                                     resample_streamlines=(not hyperparams['keep_step_size']) and train_mode,
                                                     feed_previous_direction=hyperparams['feed_previous_direction'])
    else:
        raise ValueError("Unknown model!")
