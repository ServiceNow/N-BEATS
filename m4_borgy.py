import getpass
import os

borgy_args = [
    "--image=images.borgy.elementai.lan/nbeats/base:%s" % getpass.getuser(),
    "-v", "%s:/project" % os.path.basename(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))),
    "-w", "/project/sources",
    "-e", "PYTHONPATH=/project/sources",
    "--cpu=1",
    "--gpu=1",
    "--mem=16",
    "--restartable"
]