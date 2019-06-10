import argparse
import getpass
import os
from subprocess import Popen

image_name = 'images.borgy.elementai.net/nbeats:%s' % getpass.getuser()
source_directory = os.path.dirname(os.path.realpath(__file__))
project_directory = os.path.dirname(source_directory)


def build_and_push():
    Popen(['docker', 'build', '.', '-t', image_name]).wait()
    Popen(['docker', 'push', image_name]).wait()


def submit_to_borgy(experiments_dir_name):
    borgy_args = [
        "--image=%s" % image_name,
        "-v", "%s:/project" % os.path.basename(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))),
        "-w", "/project/sources",
        "-e", "PYTHONPATH=/project/sources",
        "--cpu=1",
        "--gpu=1",
        "--mem=16",
        "--restartable"
    ]

    experiments_path = os.path.join(project_directory, 'experiments', experiments_dir_name)

    for i, experiment in enumerate(os.listdir(experiments_path)):
        experiment_path = os.path.join(experiments_path, experiment)
        print('Experiment %d: %s' % (i, experiment))
        command = 'python m4_main.py train --name %s/%s >> %s/experiment.log 2>&1' % (
            experiments_dir_name, experiment, experiment_path)
        Popen(borgy_args + ['--', 'bash', '-c', command]).wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', metavar='CMD', type=str, choices=['build',
                                                                 'train'],
                        help='Command to execute')
    parser.add_argument('--experiment', type=str, default='', help='Experiment name')

    args = parser.parse_args()

    if args.cmd == 'build':
        build_and_push()
    elif args.cmd == 'train':
        submit_to_borgy(args.experiment)
