#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D MOVING DISCS

Created on Sat Jan  2 14:52:26 2021

@author: phil
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys
sys.path.append('./../LIB/sPOD/lib/')
import numpy as np
from numpy import exp, mod,meshgrid
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sPOD_tools import shifted_rPCA, shifted_POD, build_all_frames
from Shifts import Shifts
from transforms import transforms
from scipy.optimize import basinhopping
from scipy.fft import fft
from plot_utils import show_animation, save_fig
from farge_colormaps import farge_colormap_multi


###############################################################################
cm = farge_colormap_multi()
##########################################
# %% Define your DATA:
##########################################
sourceDir = "/home/pkrah/Downloads/SnapShotMatrix558_49.npy"
dat = np.load(sourceDir)
dat = dat[:1000,:]
N = np.size(dat,0)
Nt = np.size(dat,1)
L = [1000]
x = np.linspace(0,L[0],N)
t = np.linspace(0,700,Nt)
shift = Shifts(dat,x,t)
# nmodes = 2          # reduction of singular values

data_shape = [N,1, 1, Nt]
dx = x[1] - x[0]
q = np.reshape(dat,data_shape)
# %% Create Trafo

shift_trafo_1 = transforms(data_shape, L, shifts=shift[0], dx=[dx], use_scipy_transform=True)
shift_trafo_2 = transforms(data_shape, L, shifts=shift[1], dx=[dx], use_scipy_transform=True)
shift_trafo_3 = transforms(data_shape,L, shifts = shift[0],trafo_type="identity", dx = [dx], use_scipy_transform=False )
qshift1 = shift_trafo_1.reverse(q)
qshift2 = shift_trafo_2.reverse(q)
plt.pcolormesh(np.reshape(qshift1,[N,Nt]))
plt.plot([0,Nt],[397,397],'r-')
# print("Reverse shifts done")
#
qshiftreverse = shift_trafo_1.apply(shift_trafo_1.reverse(q))
res = q - qshiftreverse
interp_err = np.linalg.norm(np.reshape(res, -1)) / np.linalg.norm(np.reshape(q, -1))
print("err =  %4.4e " % interp_err)
# plt.pcolormesh(X, Y, q[..., 0, 0] - qshiftreverse[..., 0, 0], cmap=cm)
# plt.colorbar()
"""
# %% Test Trafo

# figs,axs = plt.subplots(3,1,sharex=True)
# axs[0].pcolormesh(X,Y,qshift1[...,0,0])
# axs[1].pcolormesh(X,Y,qshift2[...,0,0])
# axs[2].pcolormesh(X,Y,q[...,0,0]-qshiftreverse[...,0,0])
# axs[0].set_title(r"$q^1$ Frame 1")
# axs[1].set_title(r"$q^2$ Frame 2")
# axs[2].set_title(r"$q - T^s_1 q^1 + T^s_2 q^2$ Data")
# for it in range(Nt):
#     axs[0].pcolormesh(X,Y,qshift1[...,0,it])
#     axs[1].pcolormesh(X,Y,qshift2[...,0,it])
#     axs[2].pcolormesh(X,Y,q[...,0,it]-qshiftreverse[...,0,it])
#     plt.show()
#     plt.pause(0.001)
"""
# %% Run shifted POD
trafos= [shift_trafo_1, shift_trafo_2,shift_trafo_3]
qmat = np.reshape(q, [-1, Nt])
#ret = shifted_POD(qmat, trafos, nmodes=[1,1,1], eps=interp_err, Niter=100, visualize=True)

[N,M]= np.shape(qmat)
mu0 = N * M / (4 * np.sum(np.abs(qmat)))*0.0001
lambd0 = 1 / np.sqrt(np.maximum(M, N))*10
ret = shifted_rPCA(qmat, trafos, eps =interp_err, Niter=100, visualize=True, mu=mu0, lambd= lambd0)
qframes, qtilde = ret.frames, ret.data_approx
qtilde = np.reshape(qtilde, data_shape)
# %%
fig,ax = plt.subplots(1,2,num=2)
vmin = np.min(q)
vmax = np.max(q)
ax[0].pcolormesh(np.reshape(qtilde,[-1,Nt]),vmin=vmin,vmax=vmax)
ax[1].pcolormesh(np.reshape(q,[-1,Nt]),vmin=vmin,vmax=vmax)
fig,ax = plt.subplots(1,3,num=3)
ax[1].pcolormesh(np.reshape(qframes[0].build_field(),[-1,Nt]),vmin=vmin,vmax=vmax)
ax[0].pcolormesh(np.reshape(qframes[1].build_field(),[-1,Nt]),vmin=vmin,vmax=vmax)
ax[2].pcolormesh(np.reshape(qframes[2].build_field(),[-1,Nt]),vmin=vmin,vmax=vmax)
