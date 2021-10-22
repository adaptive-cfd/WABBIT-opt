
import sys
import numpy as np
from numpy import exp, mod,meshgrid
from numpy.linalg import norm
from utils import bin_array
from wabbit_utils import run_wabbit




D = 3
mu_max = 8.2
# mu_vecs = np.concatenate([mu_max*np.eye(Nt,D), -mu_max*np.eye(Nt,D)],axis=1)
mu_vecs = np.asarray([(bin_array(i, D) * 2 - np.ones(D)) * mu_max for i in range(2 ** D)]).T

## loop over the different parameters and execute wabbit
mpicommand = "mpirun -np 216 --hostfile=hosts"
memory = "--memory=2GB"

for mu in mu_vecs.T:
    value_str = ' '.join(map(str,mu))+';'
    params_dict = {"inifiles":["kin1.ini","kin2.ini"],"section": "Wingsection", "key": "ai_y0", "value": value_str}
    run_wabbit(params_dict,params_inifile="two_moving_cylinders.ini",mpicommand= mpicommand, memory=memory)