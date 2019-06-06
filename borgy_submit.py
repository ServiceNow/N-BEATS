import argparse
import getpass
import os
from subprocess import Popen

borgy_args = [
    "--image=images.borgy.elementai.lan/m4/base:%s" % getpass.getuser(),
    "-v", "%s:/project" % os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir)),
    "-w", "/project/sources",
    "-e", "PYTHONPATH=/project/sources",
    "--cpu=1",
    "--gpu=1",
    "--mem=16",
    "--restartable"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default='', help='Experiment name')

    args = parser.parse_args()

    process_list = []
    for experiment in os.listdir(args.path):
        experiment_full_path = os.path.join(args.path, experiment)
        experiment_name = '%s/%s' % (os.path.basename(args.path), experiment)
        cmd = ['borgy', 'submit', '--name', "%s" % experiment_name] + borgy_args + ['--',
                                                                                     'python',
                                                                                     'm4_main.py',
                                                                                     'train',
                                                                                     '--name',
                                                                                     experiment_name]
        str_cmd = ' '.join(['"' + arg + '"' for arg in cmd])
        print(str_cmd)

        with open(os.path.join(experiment_full_path, 'borgy_submit.cmd'), 'w') as fd:
            fd.write(str_cmd)

        process = Popen(cmd)
        process_list.append(process)

    for process in process_list:
        process.wait()
