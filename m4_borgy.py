import argparse
import getpass
import os
from subprocess import Popen

from m4.settings import M4_EXPERIMENTS_DIR

image_name = 'images.borgy.elementai.net/nbeats:%s' % getpass.getuser()

source_directory = os.path.dirname(os.path.realpath(__file__))
project_directory = os.path.dirname(source_directory)

container_project_path = os.path.join(os.sep, 'project')
container_source_path = os.path.join(container_project_path, 'source')
log_file_name = 'experiment.log'


def build_and_push():
    Popen(['docker', 'build', '.', '-t', image_name]).wait()
    Popen(['docker', 'push', image_name]).wait()


def submit_to_borgy(experiments_dir_name):
    borgy_args = [
        '--image=%s' % image_name,
        '-v', '%s:%s' % (project_directory, container_project_path),
        '-w', container_source_path,
        '-e', 'PYTHONPATH=%s' % container_source_path,
        '--cpu=1',
        '--gpu=1',
        '--mem=16',
        '--restartable'
    ]
    experiments_path = os.path.join(project_directory, 'experiments', experiments_dir_name)

    for i, experiment in enumerate(os.listdir(experiments_path)):
        print('Experiment %d: %s' % (i, experiment))
        command = 'python m4_main.py train --name %s/%s >> %s 2>&1' % (
            experiments_dir_name, experiment,
            os.path.join(M4_EXPERIMENTS_DIR, experiments_dir_name, experiment, log_file_name))
        print(' '.join(
            ['borgy', 'submit', '--name=%s/%s' % (experiments_dir_name, experiment)] + borgy_args + ['--', 'bash', '-c',
                                                                                                     command]))
        Popen(
            ['borgy', 'submit', '--name=%s/%s' % (experiments_dir_name, experiment)] + borgy_args + ['--', 'bash', '-c',
                                                                                                     command]).wait()


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
