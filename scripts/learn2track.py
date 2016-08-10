#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import argparse

import theano.tensor as T

from smartlearner import Trainer
from smartlearner import tasks, views, stopping_criteria
from smartlearner.direction_modifiers import DirectionClipping


from learn2track import utils
from learn2track.utils import Timer
from learn2track.factories import WEIGHTS_INITIALIZERS, weigths_initializer_factory
from learn2track.factories import optimizer_factory
from learn2track.factories import model_factory
from learn2track.factories import loss_factory

from learn2track.batch_schedulers import StreamlinesBatchSchedulerMultiSubjects


def build_train_gru_argparser(subparser):
    DESCRIPTION = "Train a GRU."

    p = subparser.add_parser("gru_regression", description=DESCRIPTION, help=DESCRIPTION)

    # p.add_argument('dataset', type=str, help='folder containing training data (.npz files).')

    # Model options (GRU)
    model = p.add_argument_group("GRU arguments")

    model.add_argument('--hidden-sizes', type=int, nargs='+', default=500,
                       help="Size of the hidden layers. Default: 500")

    model.add_argument('--weights-initialization', type=str, default='orthogonal', choices=WEIGHTS_INITIALIZERS,
                       help='which type of initialization to use when creating weights [{0}].'.format(", ".join(WEIGHTS_INITIALIZERS)))
    model.add_argument('--initialization-seed', type=int, default=1234,
                       help='seed used to generate random numbers. Default=1234')

    model.add_argument('--learn-to-stop', action="store_true",
                       help='if specified, the model will be trained to learn when to stop tracking')

    # General parameters (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')


def build_argparser():
    DESCRIPTION = ("Script to train a GRU model from a dataset of streamlines coordinates and a DWI on a regression task.")
    p = argparse.ArgumentParser(description=DESCRIPTION)

    # Dataset options
    dataset = p.add_argument_group("Data options")
    dataset.add_argument('--train-subjects', nargs='+', required=True,
                         help='folder(s) containing `dwi.nii.gz` and `dataset.npz` that will be used as training data.')
    dataset.add_argument('--valid-subjects', nargs='+', required=True,
                         help='folder(s) containing `dwi.nii.gz` and `dataset.npz` that will be used as validation data.')

    dataset.add_argument('--dwi-name', default='dwi.nii.gz',
                         help='name of the dwi file to look for in each subject folder (.nii|.nii.gz). Default: `%(default)s`')
    dataset.add_argument('--dataset-name', default='dataset.npz',
                         help='name of the dataset file to look for in each subject folder (.npz). Default: `%(default)s`')

    duration = p.add_argument_group("Training duration options")
    duration.add_argument('--max-epoch', type=int, metavar='N', default=100,
                          help='if specified, train for a maximum of N epochs. Default: %(default)s')
    duration.add_argument('--lookahead', type=int, metavar='K', default=10,
                          help='use early stopping with a lookahead of K. Default: %(default)s')
    duration.add_argument('--lookahead-eps', type=float, default=1e-3,
                          help='in early stopping, an improvement is whenever the objective improve of at least `eps`. Default: %(default)s',)

    # Training options
    training = p.add_argument_group("Training options")
    training.add_argument('--batch-size', type=int,
                          help='size of the batch to use when training the model. Default: 100.', default=100)
    # training.add_argument('--neighborhood-patch', type=int, metavar='N',
    #                       help='if specified, patch (as a cube, i.e. NxNxN) around each streamlines coordinates will be concatenated to the input.')
    training.add_argument('--noisy-streamlines-sigma', type=float,
                          help='if specified, it is the standard deviation of the gaussian noise added independently to every point of every streamlines at each batch.')
    training.add_argument('--clip-gradient', type=float,
                          help='if provided, gradient norms will be clipped to this value (if it exceeds it).')
    training.add_argument('--seed', type=int, default=1234,
                          help='seed used to generate random numbers in the batch scheduler. Default=1234')

    # Optimizer options
    optimizer = p.add_argument_group("Optimizer (required)")
    optimizer = optimizer.add_mutually_exclusive_group(required=True)
    optimizer.add_argument('--SGD', metavar="LR", type=str, help='use SGD with constant learning rate for training.')
    optimizer.add_argument('--AdaGrad', metavar="LR [EPS=1e-6]", type=str, help='use AdaGrad for training.')
    optimizer.add_argument('--Adam', metavar="[LR=0.0001]", type=str, help='use Adam for training.')
    optimizer.add_argument('--RMSProp', metavar="LR", type=str, help='use RMSProp for training.')
    optimizer.add_argument('--Adadelta', action="store_true", help='use Adadelta for training.')

    # General options (optional)
    general = p.add_argument_group("General arguments")
    general.add_argument('--name', type=str,
                         help='name of the experiment. Default: name is generated from arguments.')

    general.add_argument('-f', '--force', action='store_true', help='restart training from scratch instead of resuming.')
    general.add_argument('--view', action='store_true', help='display learning curves.')

    subparser = p.add_subparsers(title="Models", dest="model")
    subparser.required = True   # force 'required' testing
    build_train_gru_argparser(subparser)

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    print(args)

    hyperparams_to_exclude = ['max_epoch', 'force', 'name', 'view']
    experiment_path, hyperparams, resuming = utils.maybe_create_experiment_folder(args, exclude=hyperparams_to_exclude)
    print("Resuming:" if resuming else "Creating:", experiment_path)

    with Timer("Loading dataset", newline=True):
        trainset = utils.load_streamlines_datasets(args.train_subjects, args.dwi_name, args.dataset_name)
        validset = utils.load_streamlines_datasets(args.valid_subjects, args.dwi_name, args.dataset_name)
        print("Datasets:", len(trainset), " |", len(validset))

        batch_scheduler = StreamlinesBatchSchedulerMultiSubjects(trainset,
                                                                 batch_size=args.batch_size,
                                                                 noisy_streamlines_sigma=args.noisy_streamlines_sigma,
                                                                 seed=args.seed)
        print ("An epoch will be composed of {} updates.".format(batch_scheduler.nb_updates_per_epoch))
        print (batch_scheduler.input_size, args.hidden_sizes, batch_scheduler.target_size)

    with Timer("Creating model"):
        model = model_factory(hyperparams, batch_scheduler)
        model.initialize(weigths_initializer_factory(args.weights_initialization,
                                                     seed=args.initialization_seed))

    with Timer("Building optimizer"):
        loss = loss_factory(hyperparams, model, trainset)

        if args.clip_gradient is not None:
            loss.append_gradient_modifier(DirectionClipping(threshold=args.clip_gradient))

        optimizer = optimizer_factory(hyperparams, loss)

    with Timer("Building trainer"):
        trainer = Trainer(optimizer, batch_scheduler)

        # Log training error
        loss_monitor = views.MonitorVariable(loss.loss)
        avg_loss = tasks.AveragePerEpoch(loss_monitor)
        trainer.append_task(avg_loss)

        # Print average training loss.
        trainer.append_task(tasks.Print("Avg. training loss:         : {}", avg_loss))

        if args.learn_to_stop:
            l2err_monitor = views.MonitorVariable(T.mean(loss.mean_sqr_error))
            avg_l2err = tasks.AveragePerEpoch(l2err_monitor)
            trainer.append_task(avg_l2err)

            crossentropy_monitor = views.MonitorVariable(T.mean(loss.cross_entropy))
            avg_crossentropy = tasks.AveragePerEpoch(crossentropy_monitor)
            trainer.append_task(avg_crossentropy)

            trainer.append_task(tasks.Print("Avg. training L2 err:       : {}", avg_l2err))
            trainer.append_task(tasks.Print("Avg. training stopping:     : {}", avg_crossentropy))
            trainer.append_task(tasks.Print("L2 err : {0:.4f}", l2err_monitor, each_k_update=100))
            trainer.append_task(tasks.Print("stopping : {0:.4f}", crossentropy_monitor, each_k_update=100))

        # Print NLL mean/stderror.
        # train_loss = L2DistanceForSequences(model, trainset)
        # train_batch_scheduler = StreamlinesBatchScheduler(trainset, batch_size=1000,
        #                                                   noisy_streamlines_sigma=None,
        #                                                   nb_updates_per_epoch=None,
        #                                                   seed=1234)

        # train_error = views.LossView(loss=train_loss, batch_scheduler=train_batch_scheduler)
        # trainer.append_task(tasks.Print("Trainset - Error        : {0:.2f} | {1:.2f}", train_error.sum, train_error.mean))

        valid_loss = loss_factory(hyperparams, model, validset)
        valid_batch_scheduler = StreamlinesBatchSchedulerMultiSubjects(validset,
                                                                       batch_size=1000,
                                                                       noisy_streamlines_sigma=None,
                                                                       seed=1234)

        valid_error = views.LossView(loss=valid_loss, batch_scheduler=valid_batch_scheduler)
        trainer.append_task(tasks.Print("Validset - Error        : {0:.2f} | {1:.2f}", valid_error.sum, valid_error.mean))

        lookahead_loss = valid_error.sum

        direction_norm = views.MonitorVariable(T.sqrt(sum(map(lambda d: T.sqr(d).sum(), loss.gradients.values()))))
        # trainer.append_task(tasks.Print("||d|| : {0:.4f}", direction_norm))

        # logger = tasks.Logger(train_error.mean, valid_error.mean, valid_error.sum, direction_norm)
        logger = tasks.Logger(valid_error.mean, valid_error.sum, direction_norm)
        trainer.append_task(logger)

        if args.view:
            import pylab as plt

            def _plot(*args, **kwargs):
                plt.figure(1)
                plt.clf()
                plt.show(False)
                plt.subplot(121)
                plt.plot(np.array(logger.get_variable_history(0)).flatten(), label="Train")
                plt.plot(np.array(logger.get_variable_history(1)).flatten(), label="Valid")
                plt.legend()

                plt.subplot(122)
                plt.plot(np.array(logger.get_variable_history(3)).flatten(), label="||d'||")
                plt.draw()

            trainer.append_task(tasks.Callback(_plot))

        # Callback function to stop training if NaN is detected.
        def detect_nan(obj, status):
            if np.isnan(model.parameters[0].get_value().sum()):
                print("NaN detected! Not saving the model. Crashing now.")
                sys.exit()
        trainer.append_task(tasks.Callback(detect_nan, each_k_update=1))

        # Early stopping with a callback for saving every time model improves.
        def save_training(obj, status):
            """ Save training progression. """
            if np.isnan(model.parameters[0].get_value().sum()):
                print("NaN detected! Not saving the model. Crashing now.")
                sys.exit()

            print("*** Best epoch: {0} ***\n".format(obj.best_epoch))
            trainer.save(experiment_path)

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())
        trainer.append_task(tasks.PrintTime(each_k_update=100))  # Profiling

        # Add stopping criteria
        trainer.append_task(stopping_criteria.MaxEpochStopping(args.max_epoch))
        early_stopping = stopping_criteria.EarlyStopping(lookahead_loss, lookahead=args.lookahead, eps=args.lookahead_eps, callback=save_training)
        trainer.append_task(early_stopping)

    with Timer("Compiling Theano graph"):
        trainer.build_theano_graph()

    if resuming:
        with Timer("Loading"):
            trainer.load(experiment_path)

    with Timer("Training"):
        trainer.train()

    trainer.save(experiment_path)
    model.save(experiment_path)

if __name__ == "__main__":
    main()