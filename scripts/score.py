import argparse
import os
import json
from os.path import join as pjoin
import nibabel as nib
import subprocess
import textwrap
# from subprocess import check_call


def save_dict_to_json_file(path, dictionary):
    """ Saves a dict in a json formatted file. """
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def buildArgsParser():
    description = textwrap.dedent("""
        Score a tractogram using the Tractometer.
    """)

    p = argparse.ArgumentParser(description=description)
    p.add_argument('tractogram', help='tractogram file (.tck).')
    p.add_argument('test_subject',
                   help='folder containing scoring data (eg. "ismrm15_challenge/scoring_data/" or "hcp/subjects/100307/scoring_data/"')
    p.add_argument('--output', default='tractometer',
                   help='folder that will contain tractometer output (eg. segmented bundles and scores). Default: %(default)s')
    p.add_argument('--ismrm-tractometer', action="store_true",
                   help="if specified, JC's version of the Tractometer will be used. It is the same as for the ISMRM Challenge")
    return p


if __name__ == "__main__":
    parser = buildArgsParser()
    args = parser.parse_args()

    tractogram_name, ext = os.path.splitext(os.path.basename(args.tractogram))
    if ext != ".tck":
        raise ValueError("Only supporting TCK file at the moment.")

    tractometer_output = pjoin(args.output, tractogram_name)
    scoring_data = pjoin(args.test_subject, 'scoring_data')
    if not os.path.isdir(scoring_data):
        scoring_data = args.test_subject  # Already scoring_data folder

    gt_bundles_attributes_json = pjoin(scoring_data, 'gt_bundles_attributes.json')

    # Create output folder, if needed.
    try:
        os.makedirs(tractometer_output)
    except:
        pass

    # Create attributes.json
    attributes_json = pjoin(tractometer_output, "attributes.json")
    attributes = {'orientation': 'RAS',
                  'count': len(nib.streamlines.load(args.tractogram).tractogram)}
    print("Saving {}".format(attributes_json))
    save_dict_to_json_file(attributes_json, attributes)

    # Run Tractometer scoring.
    if args.ismrm_tractometer:
        cmd = ["python", "~/research/src/tractometer/scripts/score_ismrm.py",
               args.tractogram, scoring_data, attributes_json,
               gt_bundles_attributes_json, tractometer_output, "5",
               "-v", "--save_tracts", "--save_vb", "--save_ib"]
    else:
        cmd = ["python", "~/research/src/learn2track/scripts/tractometer/score_learn2track.py",
               args.tractogram, scoring_data, attributes_json,
               gt_bundles_attributes_json, tractometer_output, "5",
               "-v", "--save_tracts", "--save_vb", "--save_ib", "-f"]

    print(" ".join(cmd))
    subprocess.call(" ".join(cmd), shell=True)
