
import sys
import numpy as np
from numpy import exp, mod,meshgrid
from numpy.linalg import norm
from utils import bin_array
from wabbit_utils import run_wabbit
import time





D = 3
#mu_max = 8.2
mu_max = 4.0
# mu_vecs = np.concatenate([mu_max*np.eye(Nt,D), -mu_max*np.eye(Nt,D)],axis=1)
mu_vecs = np.asarray([(bin_array(i, D) * 2 - np.ones(D)) * mu_max for i in range(2 ** D)]).T
#mu_vecs = np.delete(mu_vecs,[0,1,2,3,4,5,6],1)
print(mu_vecs)
## loop over the different parameters and execute wabbit
#mpicommand = "mpirun -np 176 --hostfile=hosts"
mpicommand = "mpirun -np 6"

memory = "--memory=2GB"

for i,mu in enumerate(mu_vecs.T):
    value_str = ' '.join(map(str,mu))+';'
    params_dict = {"inifiles":["kin1.ini","kin2.ini"],"section": "Wingsection", "key": "ai_y0", "value": value_str}

    ## run FOM
    print("\n(%d) Running FOM at mu = "%i + value_str)
    t_cpu = time.time()
    success = run_wabbit(params_dict,params_inifile="two_moving_cylinders.ini",mpicommand= mpicommand, memory=memory)
    t_cpu = time.time() - t_cpu

    if success == 0 :
        print("FOM-simulation successfull tcpu = %2.2f!" % t_cpu )
    else:
        print("FOM-simulation broke tcpu = %2.2f!" % t_cpu)
        print("I am stopping here!")
        break
