import numpy as np
import subprocess
import sys

def load_data(fp):
    data = np.loadtxt(fp)
    X, y = data[:, :-1], data[:, -1]
    return X, y

def upload(src, dst):
    subprocess.check_call(['gsutil', 'cp', src, dst],
        stderr=sys.stdout)