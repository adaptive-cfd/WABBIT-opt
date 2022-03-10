#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 09:11:12 2018

@author: philipp krah
"""

###############################################################################
# IMPORTED MODULES
###############################################################################
import sys

sys.path.append('./../LIB/sPOD/lib/')
import numpy as np
from numpy import exp, mod, meshgrid
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sPOD_tools import shifted_rPCA, shifted_POD
from transforms import transforms
from scipy.optimize import basinhopping
from scipy.fft import fft

###############################################################################

##########################################
# %% Define your DATA:
##########################################
# plt.close("all")
Nx = 400  # number of grid points in x
Nt = 200  # numer of time intervalls

T = 0.5  # total time
L = 1  # total domain size
sigma = 0.015 * L  # standard diviation of the puls
nmodes = 1  # reduction of singular values
x = np.arange(0, Nx) / Nx * L
t = np.arange(0, Nt) / Nt * T
dx = x[1] - x[0]
dt = t[1] - t[0]
c = 1
[X, T] = meshgrid(x, t)
X = X.T
T = T.T
fun = lambda x, shifts, t: -exp(-(mod((x + shifts[0]), L) - 0.1) ** 2 / sigma ** 2) + \
                           exp(-(mod((x + shifts[1]), L) - 0.9) ** 2 / sigma ** 2)

# Define your field as a list of fields:
# For example the first element in the list can be the density of
# a flow quantity and the second element could be the velocity in 1D
shifts = [np.asarray([-c * t]), np.asarray([c * t])]
# e = np.zeros_like(t)
# e[1]=0.2*L
# e[2]=0.1*L
# e[0]=-np.sum(e[1:])
# s = np.asarray([fft(e).real])
# s[0,0]=0
# shifts = [s,-s]

density = fun(X, shifts, T)
velocity = fun(X, shifts, T)
fields = [density]  # , velocity]

#######################################
# %% CALL THE SPOD algorithm
######################################
qmat = np.reshape(fields, [Nx, Nt])
data_shape = [Nx, 1, 1, Nt]
trafos = [transforms(data_shape, [L], shifts=shifts[0], dx=[dx], use_scipy_transform=True),
          transforms(data_shape, [L], shifts=shifts[1], dx=[dx], use_scipy_transform=True)]

mu = Nx * Nt / (4 * np.sum(np.abs(qmat))) * 0.1
ret = shifted_rPCA(qmat, trafos, nmodes_max=np.max(nmodes) + 10, eps=1e-16, Niter=500, use_rSVD=True, mu=mu, lambd=1e14)
sPOD_frames, qtilde, rel_err = ret.frames, ret.data_approx, ret.rel_err_hist
# sPOD_frames, qtilde  = sPOD_distribute_residual(qmat, trafos, nmodes, eps=1e-16, Niter=400, use_rSVD = False)
###########################################
# %% results sPOD frames
##########################################
# the result is a list of the decomposed field.
# each element of the list contains a frame of the decomposition.<
# If you want to plot the k-th frame use:
# 1. frame
k_frame = 0
plt.figure(num=10)
plt.subplot(121)
plt.pcolormesh(X, T, sPOD_frames[k_frame].build_field())
plt.suptitle("sPOD Frames")
plt.xlabel(r'$N_x$')
plt.ylabel(r'$N_t$')
plt.title(r"$q^" + str(k_frame) + "(x,t)$")
# 2. frame
k_frame = 1
plt.subplot(122)
plt.pcolormesh(X, T, sPOD_frames[k_frame].build_field())
plt.xlabel(r'$N_x$')
plt.ylabel(r'$N_t$')
plt.title(r"$q^" + str(k_frame) + "(x,t)$")
# by default this will plot the field in the first component
# of your field list (here: density)

###########################################
# relative error
##########################################

plt.figure(5)
plt.semilogy(rel_err)
plt.title("relative error")
plt.ylabel(r"$\frac{||X - \tilde{X}_i||_2}{||X||_2}$")
plt.xlabel(r"$i$")

###########################################
# %% define optimization goal
##########################################
Wt = np.zeros([Nt, 1])
Wt[Nt // 2] = 1

opt_goal = lambda Qmat: norm(Qmat @ Wt, ord='fro') ** 2 * dx * dt


def my_interpolated_state(sPOD_frames, time, mu):
    shiftsnew = [np.asarray([-mu * time]), np.asarray([mu * time])]
    qtilde = 0
    for shift, frame in zip(shiftsnew, sPOD_frames):
        trafo = transforms(frame.data_shape, frame.trafo.domain_size, shifts=shift, dx=frame.trafo.dx,
                           use_scipy_transform=True)
        qtilde += trafo.apply(frame.build_field())

    return qtilde


opt_fun = lambda mu: opt_goal(np.reshape(my_interpolated_state(sPOD_frames, t, mu), [Nx, Nt]))

###########################################
# %% optimize with global optimizer
##########################################

bounds = lambda **kwargs: kwargs["x_new"] < 1.8 * c and kwargs["x_new"] > 0 * c
ret = basinhopping(opt_fun, 1 * c, accept_test=bounds, seed=1)
mu_star = ret.x
print("global minimum: mu_* = %.4f, J(mu_*) = %.4e" % (ret.x, ret.fun))

###########################################
# %% plot results of optimization
##########################################
fig, ax = plt.subplots(num=100)
parameter_list = np.linspace(1.4 * c, 1.8 * c, 50)
J_list = [opt_fun(mu) for mu in parameter_list]

plt.plot(parameter_list, J_list)
plt.vlines(mu_star, *ax.get_ylim(), color='r', linestyles='dashed')
plt.text(mu_star, ax.get_ylim()[0], r'$\mu^*$', color="r")
plt.xlabel(r"parameter $\mu$")
plt.ylabel(r"functional $J(\mu)$")
plt.show()

fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].pcolormesh(X, T, qmat, shading='nearest')
ax[0].set_title("data")
plt.xlabel(r" $x$")
plt.ylabel(r"time $t$")
ax[1].pcolormesh(X, T, my_interpolated_state(sPOD_frames, t, mu_star), shading='nearest')
ax[1].set_title(r"optimal $\mu=\mu_*$")
plt.xlabel(r" $x$")
plt.ylabel(r"time $t$")
plt.show()
