import os
from scipy.io import loadmat




def read_ACM_dat(path, sample_fraction = 1):

    data = loadmat(path)
    fields = data["data"]
    mask = fields[0, ...].T
    p = fields[1, ...].T
    ux = fields[2, ...].T
    uy = fields[3, ...].T
    time = data["time"].flatten()
    time = time - time[0]

    Ngrid = [fields.shape[2] // sample_fraction, fields.shape[3] // sample_fraction]
    domain_size = data['domain_size'][0]
    dx = data["dx"][0]*sample_fraction
    frac = sample_fraction
    return ux[::frac,::frac,:], uy[::frac,::frac,:], p[::frac,::frac,:], mask[::frac,::frac,:], time, Ngrid, dx, domain_size