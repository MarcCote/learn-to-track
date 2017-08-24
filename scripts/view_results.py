#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import argparse
import csv
import os
import pickle
from collections import OrderedDict
from os.path import join as pjoin

import numpy as np
from smartlearner.utils import load_dict_from_json_file

DESCRIPTION = 'Gather experiments results and save them in a CSV file.'


def buildArgsParser():
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('names', type=str, nargs='+', help='name/path of the experiments.')
    p.add_argument('--out', default="results.csv", help='save table in a CSV file. Default: results.csv')
    p.add_argument('-v', '--verbose', action="store_true", help='verbose mode')
    return p


class Experiment(object):
    def __init__(self, tractometer_path):  # , tractography_name):
        self.tractometer_path = tractometer_path
        self.tractogram_name = os.path.basename(tractometer_path)

        self.experiment_path = os.path.normpath(pjoin(self.tractometer_path, "..", ".."))
        self.experiment_name = os.path.basename(self.experiment_path)

        self.results_file = pjoin(self.experiment_path, "results.json")
        self.tractometer_scores_file = pjoin(self.tractometer_path, "scores", "{}.json".format(self.tractogram_name))
        self.hyperparams_file = pjoin(self.experiment_path, "hyperparams.json")
        self.status_file = pjoin(self.experiment_path, "training", "status.json")
        self.early_stopping_file = self._find_task("early_stopping")

        self.results = {}
        if os.path.isfile(self.results_file):
            self.results = load_dict_from_json_file(self.results_file)

        self.hyperparams = load_dict_from_json_file(self.hyperparams_file)
        self.status = load_dict_from_json_file(self.status_file)

        self.tractometer_scores = {}
        if os.path.isfile(self.tractometer_scores_file):
            self.tractometer_scores = load_dict_from_json_file(self.tractometer_scores_file)
        elif os.path.isfile(self.tractometer_scores_file[:-5] + '.pkl'):
            self.tractometer_scores = pickle.load(open(self.tractometer_scores_file[:-5] + '.pkl', 'rb'))
        else:
            print("No tractometer results yet for: {}".format(self.tractometer_scores_file))

        self.early_stopping = {}
        if os.path.isfile(self.early_stopping_file):
            self.early_stopping = load_dict_from_json_file(self.early_stopping_file)

    def _find_task(self, task_name):
        """ Search all task folders for task_name.json

        :param task_name: (str) Name of the task
        :return: (str) Path of the .json task file or None if not found
        """
        tasks_path = pjoin(self.experiment_path, "training", "tasks")
        for task_folder in os.listdir(tasks_path):
            file_path = pjoin(tasks_path, task_folder, task_name + ".json")
            if os.path.isfile(file_path):
                return file_path
        return None


def list_of_dict_to_csv_file(csv_file, list_of_dicts):
    keys = list_of_dicts[0].keys()
    with open(csv_file, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(list_of_dicts)


def get_optimizer(e):
    if e.hyperparams.get("SGD") is not None:
        return "SGD"
    elif e.hyperparams.get("AdaGrad") is not None:
        return "AdaGrad"
    elif e.hyperparams.get("Adam") is not None:
        return "Adam"
    elif e.hyperparams.get("RMSProp") is not None:
        return "RMSProp"
    elif e.hyperparams.get("Adadelta") is not None:
        return "Adadelta"

    return ""

def get_model(e):
    model = e.hyperparams.get("model")
    if model == "gru_mixture":
        model += "_{}".format(e.hyperparams.get("n_gaussians"))
    return model


def extract_L2_error(results, dataset, metric):
    if dataset not in results:
        return ""

    return str(results[dataset][metric])


def extract_result_from_experiment(e):
    """e: `Experiment` object"""
    entry = OrderedDict()
    entry["Model"] = get_model(e)
    entry["Experiment"] = e.experiment_name[:6]
    entry["Tractogram"] = e.tractogram_name
    entry["Seed"] = e.hyperparams.get("seed", "")
    entry["Model seed"] = e.hyperparams.get("initialization_seed", "")
    entry["Hidden Size(s)"] = "-".join(map(str, e.hyperparams.get("hidden_sizes", [])))
    entry["Feed previous direction"] = e.hyperparams.get("feed_previous_direction", "")
    entry["Predict offset"] = e.hyperparams.get("predict_offset", "")
    entry["Use layer normalization"] = e.hyperparams.get("use_layer_normalization", "")
    entry["Use sh coeffs"] = e.hyperparams.get("use_sh_coeffs", "")
    entry["Noise sigma"] = e.hyperparams.get("noisy_streamlines_sigma", "")
    entry["Drop prob"] = e.hyperparams.get("drop_prob", "")
    entry["Zoneout"] = e.hyperparams.get("use_zoneout", "")
    entry["Skip connections"] = e.hyperparams.get("skip_connections", "")
    entry["Best Epoch"] = e.early_stopping.get("best_epoch", "")
    entry["Max Epoch"] = e.status.get("current_epoch", "")

    # Results
    error_type = "EV_L2_error"
    if e.hyperparams["model"] in ['gru_gaussian', 'gru_mixture', 'gru_multistep']:
        error_type = "NLL"

    entry["Train error"] = extract_L2_error(e.results, "trainset_{}".format(error_type), "mean")
    entry["Valid error"] = extract_L2_error(e.results, "validset_{}".format(error_type), "mean")

    # Tractometer results
    entry["VC"] = str(e.tractometer_scores.get("VC", "0"))
    entry["IC"] = str(e.tractometer_scores.get("IC", "0"))
    entry["NC"] = str(e.tractometer_scores.get("NC", "0"))
    entry["VB"] = str(e.tractometer_scores.get("VB", "0"))
    entry["IB"] = str(e.tractometer_scores.get("IB", "0"))
    entry["count"] = str(e.tractometer_scores.get("total_streamlines_count", ""))

    overlap_per_bundle = e.tractometer_scores.get("overlap_per_bundle", {})
    overreach_per_bundle = e.tractometer_scores.get("overreach_per_bundle", {})
    entry["Avg. Overlap"] = str(np.mean(list(map(float, overlap_per_bundle.values()))))
    entry["Avg. Overreach"] = str(np.mean(list(map(float, overreach_per_bundle.values()))))

    entry["VCCR"] = ""
    if len(e.tractometer_scores) > 0:
        try:
            entry["VCCR"] = str(float(entry["VC"]) / (float(entry["VC"]) + float(entry["IC"])))
        except ZeroDivisionError:
            print("A ZeroDivisionError happened when computing VCCR: VC={}; IC={}".format(entry["VC"], entry["IC"]))

    # Other hyperparameters
    entry["Optimizer"] = get_optimizer(e)
    entry["Optimizer params"] = e.hyperparams.get(get_optimizer(e), "")
    entry["Clip Gradient"] = e.hyperparams.get("clip_gradient", "")
    entry["Batch Size"] = e.hyperparams.get("batch_size", "")
    entry["Weights Initialization"] = e.hyperparams.get("weights_initialization", "")
    entry["Look Ahead"] = e.hyperparams.get("lookahead", "")
    entry["Look Ahead eps"] = e.hyperparams.get("lookahead_eps", "")

    streamlines_per_bundle = e.tractometer_scores.get("streamlines_per_bundle", {})
    entry['CA'] = str(streamlines_per_bundle.get("CA", "0"))
    entry['CC'] = str(streamlines_per_bundle.get("CC", "0"))
    entry['CP'] = str(streamlines_per_bundle.get("CP", "0"))
    entry['CST_left'] = str(streamlines_per_bundle.get("CST_left", "0"))
    entry['CST_right'] = str(streamlines_per_bundle.get("CST_right", "0"))
    entry['Cingulum_left'] = str(streamlines_per_bundle.get("Cingulum_left", "0"))
    entry['Cingulum_right'] = str(streamlines_per_bundle.get("Cingulum_right", "0"))
    entry['FPT_left'] = str(streamlines_per_bundle.get("FPT_left", "0"))
    entry['FPT_right'] = str(streamlines_per_bundle.get("FPT_right", "0"))
    entry['Fornix'] = str(streamlines_per_bundle.get("Fornix", "0"))
    entry['ICP_left'] = str(streamlines_per_bundle.get("ICP_left", "0"))
    entry['ICP_right'] = str(streamlines_per_bundle.get("ICP_right", "0"))
    entry['ILF_left'] = str(streamlines_per_bundle.get("ILF_left", "0"))
    entry['ILF_right'] = str(streamlines_per_bundle.get("ILF_right", "0"))
    entry['MCP'] = str(streamlines_per_bundle.get("MCP", "0"))
    entry['OR_left'] = str(streamlines_per_bundle.get("OR_left", "0"))
    entry['OR_right'] = str(streamlines_per_bundle.get("OR_right", "0"))
    entry['POPT_left'] = str(streamlines_per_bundle.get("POPT_left", "0"))
    entry['POPT_right'] = str(streamlines_per_bundle.get("POPT_right", "0"))
    entry['SCP_left'] = str(streamlines_per_bundle.get("SCP_left", "0"))
    entry['SCP_right'] = str(streamlines_per_bundle.get("SCP_right", "0"))
    entry['SLF_left'] = str(streamlines_per_bundle.get("SLF_left", "0"))
    entry['SLF_right'] = str(streamlines_per_bundle.get("SLF_right", "0"))
    entry['UF_left'] = str(streamlines_per_bundle.get("UF_left", "0"))
    entry['UF_right'] = str(streamlines_per_bundle.get("UF_right", "0"))

    # Other results
    entry["Test error"] = extract_L2_error(e.results, "testset_{}".format(error_type), "mean")
    entry["Std. Train error"] = extract_L2_error(e.results, "trainset_{}".format(error_type), "stderror")
    entry["Std. Valid error"] = extract_L2_error(e.results, "validset_{}".format(error_type), "stderror")
    entry["Std. Test error"] = extract_L2_error(e.results, "testset_{}".format(error_type), "stderror")
    entry["Std. Overlap"] = str(np.std(list(map(float, overlap_per_bundle.values()))))
    entry["Std. Overreach"] = str(np.std(list(map(float, overreach_per_bundle.values()))))

    entry["Training Time"] = e.status.get("training_time", "")
    entry["Dataset"] = os.path.basename(e.hyperparams.get("dataset", ""))
    entry["Experiment full name"] = e.experiment_name

    if "missing" in entry["Dataset"]:
        bundle_name = entry["Dataset"][:-4].split("_")[-1]
        missing_bundle_count = 0
        missing_bundle_overlap = []
        missing_bundle_overreach = []
        for k, v in streamlines_per_bundle.items():
            if k.startswith(bundle_name):
                missing_bundle_count += int(v)
                missing_bundle_overlap.append(overlap_per_bundle.get(k, 0))
                missing_bundle_overreach.append(overreach_per_bundle.get(k, 0))

        entry["Missing Bundle Count"] = str(missing_bundle_count)
        entry["Missing Bundle Overlap"] = str(np.mean(missing_bundle_overlap))
        entry["Missing Bundle Overreach"] = str(np.mean(missing_bundle_overreach))

    return entry


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    experiments_results = []

    for experiment_path in args.names:
        tractometer_folder_path = pjoin(experiment_path, "tractometer")
        try:
            for tractometer_evaluation_name in os.listdir(tractometer_folder_path):
                tractometer_evaluation_path = pjoin(tractometer_folder_path, tractometer_evaluation_name)
                scores_dir = pjoin(tractometer_evaluation_path, "scores")

                if not os.path.isdir(scores_dir):
                    continue

                try:
                    experiment = Experiment(tractometer_evaluation_path)
                    experiments_results.append(extract_result_from_experiment(experiment))
                    if args.verbose:
                        print("Fetched results for {}".format(tractometer_evaluation_name))
                except IOError as e:
                    if args.verbose:
                        print(str(e))

                    print("Skipping: '{}' for {}".format(experiment_path, tractometer_evaluation_name))
        except FileNotFoundError:
            try:
                print("No tractometer results found, loading evaluation scores...")
                experiment = Experiment(experiment_path)
                experiments_results.append(extract_result_from_experiment(experiment))
            except FileNotFoundError:
                print("Could not load experiment results from {}".format(experiment_path))

    list_of_dict_to_csv_file(args.out, experiments_results)


if __name__ == "__main__":
    main()
