import numpy as np
import sys
import matplotlib.pyplot as plt
import pathlib
import os
from sklearn.utils import extmath

sys.path.append('./../LIB/MLbasedModelReduction/POD_DL_ROM_LIB/')
sys.path.append('./../LIB/sPOD/lib/')

from TrainingFramework import TrainingFramework
from TestingFramework import TestingFramework
from transforms import transforms
from scipy.fft import fft


def snapshot_prediction(sPOD_frames, frame_amplitude_list, D, Nx, Nt, mu_vec):
    s = np.zeros(Nt)
    s[:D] = mu_vec[:D, 0]
    shifts = np.asarray([fft(s).imag])
    shiftsnew = [shifts, -shifts]

    nf = 0
    qtilde = 0
    for shift, frame, amplitudes in zip(shiftsnew, sPOD_frames, frame_amplitude_list):
        Modes = frame.modal_system["U"]
        Q = Modes[:, :frame.Nmodes] @ amplitudes
        qframe = np.reshape(Q, [Nx, 1, 1, Nt])
        trafo = transforms([Nx, 1, 1, Nt], frame.trafo.domain_size, shifts=shift, dx=frame.trafo.dx,
                           use_scipy_transform=True)
        qtilde += trafo.apply(qframe)

        nf = nf + 1

    return qtilde


if __name__ == "__main__":

    Nx = 800
    Nt = 400
    D = 4

    log_dir = "MLdata/"
    sPOD_frames = np.load(log_dir + 'frames.npy', allow_pickle=True)
    U = np.load(log_dir + 'U_matrix.npy', allow_pickle=True)
    Params_train = np.load(log_dir + 'params_train.npy', allow_pickle=True)
    Params_test = np.load(log_dir + 'params_test.npy', allow_pickle=True)
    time_amplitudes = np.load(log_dir + 'time_amplitudes.npy', allow_pickle=True)
    NumModesPerFrame = np.load(log_dir + 'NumModesPerFrame.npy', allow_pickle=True)
    snapshot_test = np.load(log_dir + 'snapshot_test.npy', allow_pickle=True)
    snapshot_test_frame = np.load(log_dir + 'snapshot_test_frame.npy', allow_pickle=True)
    snapshot_train = np.load(log_dir + 'snapshot_train.npy', allow_pickle=True)
    snapshot_train_frame = np.load(log_dir + 'snapshot_train_frame.npy', allow_pickle=True)

    dict_test = []
    nf = 0
    for frame in range(len(NumModesPerFrame)):
        Snapshot = sPOD_frames[nf].build_field()

        NumSnapshots = int(Snapshot.shape[1]) // Nt
        # Parameters needed for the training and validation of the framework
        dict_network = {
            'FOM': False,  # This switch is true for full order model input and false for only time amplitude matrix
            'snapshot_train': Snapshot,
            'snapshot_test': None,
            'time_amplitude_train': time_amplitudes[nf],
            'time_amplitude_test': None,
            'parameter_train': Params_train[nf],
            'parameter_test': Params_test[nf],
            'num_parameters': int(Params_train[nf].shape[0]),  # n_mu + 1
            'num_time_steps': Nt,  # N_t
            'num_samples': int(time_amplitudes[nf].shape[1]),  # N_train x N_t
            'num_test_samples': Params_test[nf].shape[1],  # N_test x N_t
            'batch_size': 400,
            'num_early_stop': 1500,  # Number of epochs for the early stopping
            'scaling': True,  # true if the data should be scaled
            'perform_svd': 'randomized',  # '', 'normal', 'randomized'
            'learning_rate': 0.0005,  # eta  0.001
            'full_order_model_dimension': Nx,  # N_h
            'reduced_order_model_dimension': NumModesPerFrame[frame],  # N
            'encoded_dimension': 6,  # dimension of the system after the encoder
            'num_dimension': 1,  # Number of channels (d)
            'omega_h': 0.8,
            'omega_N': 0.2,
            'typeConv': '1D'  # Type of convolutional layer for the network : '1D' or '2D'
        }

        train_model = TrainingFramework(dict_network, split=0.67)
        train_model.training(epochs=100, save_every=1, print_every=1)

        dict_test.append(dict_network)

        nf = nf + 1


    ############################################################
    # Testing Section
    testing_method = 'weight_based'
    if testing_method == 'model_based':
        log_folder_base = 'training_results_local/'
        num_frame_models = nf
        log_folder_trained_model = []
        for num_frame in range(num_frame_models):
            f = sorted(pathlib.Path(log_folder_base).glob('*/'), key=os.path.getmtime)[-(num_frame + 1)]
            log_folder_trained_model.append(f)
        log_folder_trained_model.reverse()

        time_amplitudes_predicted = []
        # Testing for each frame
        for frame, folder_name in enumerate(log_folder_trained_model):
            # Testing for each snapshot
            pt = dict_test[frame]['parameter_test']
            for sp in range(NumSnapshots):
                dict_test[frame]['parameter_test'] = pt[:, sp * Nt:(sp + 1) * Nt]
                test_model = TestingFramework(dict_test[frame])
                test_model.testing(log_folder_trained_model=str(folder_name), testing_method='model_based')
                time_amplitudes_predicted.append(test_model.time_amplitude_test_output)
    else:
        log_folder_base = 'training_results_local/'
        num_frame_models = nf
        log_folder_trained_model = []
        for num_frame in range(num_frame_models):
            f = sorted(pathlib.Path(log_folder_base).glob('*/'), key=os.path.getmtime)[-(num_frame + 1)]
            log_folder_trained_model.append(f)
        log_folder_trained_model.reverse()

        time_amplitudes_predicted = []
        # Testing for each frame
        for frame, folder_name in enumerate(log_folder_trained_model):
            # Testing for each snapshot
            pt = dict_test[frame]['parameter_test']
            for sp in range(NumSnapshots):
                dict_test[frame]['parameter_test'] = pt[:, sp * Nt:(sp + 1) * Nt]
                test_model = TestingFramework(dict_test[frame])
                test_model.testing(log_folder_trained_model=str(folder_name), testing_method='weight_based')
                time_amplitudes_predicted.append(test_model.time_amplitude_test_output)

    if dict_test[0]['FOM']:
        U_train = []
        for nf in range(num_frame_models):
            U, _, _ = extmath.randomized_svd(snapshot_train_frame[nf],
                                             n_components=dict_test[nf]['reduced_order_model_dimension'],
                                             transpose=False,
                                             flip_sign=False, random_state=123)
            U_train.append(U)

    # Plotting and analysis
    for sp in range(NumSnapshots):
        time_amplitudes_predicted_list = []
        time_amplitudes_actual_list = []
        for frame in range(num_frame_models):
            VT = np.matmul(U_train[frame].transpose(), snapshot_test_frame[frame][:, sp * Nt:(sp + 1) * Nt])
            time_amplitudes_actual_list.append(VT)
            time_amplitudes_predicted_list.append(time_amplitudes_predicted[sp + frame * NumSnapshots])
            num = np.sqrt(
                np.mean(np.linalg.norm(VT - time_amplitudes_predicted[sp + frame * NumSnapshots], 2, axis=1) ** 2))
            den = np.sqrt(np.mean(np.linalg.norm(VT, 2, axis=1) ** 2))
            print("Relative time amplitude error indicator for snapshot:{} and frame:{} is {}".format(sp, frame,
                                                                                                      num / den))
            if dict_test[frame]['FOM']:
                Q_recon = np.matmul(U_train[frame], VT)
                num = np.sqrt(np.mean(np.linalg.norm(snapshot_test_frame[frame][:, sp * Nt:(sp + 1) * Nt] - Q_recon, 2, axis=1) ** 2))
                den = np.sqrt(np.mean(np.linalg.norm(snapshot_test_frame[frame][:, sp * Nt:(sp + 1) * Nt], 2, axis=1) ** 2))
                print("Relative snapshot error indicator for reconstructed snapshot:{} and frame:{} is {}".format(sp, frame, num / den))

        qtilde = snapshot_prediction(sPOD_frames, time_amplitudes_predicted_list, D, Nx, Nt,
                                     Params_test[0][:D, sp * Nt:(sp + 1) * Nt])
        num = np.sqrt(
            np.mean(np.linalg.norm(snapshot_test[:, sp * Nt:(sp + 1) * Nt] - np.squeeze(qtilde), 2, axis=1) ** 2))
        den = np.sqrt(np.mean(np.linalg.norm(snapshot_test[:, sp * Nt:(sp + 1) * Nt], 2, axis=1) ** 2))
        print("Relative NN reconstruction error indicator for snapshot:{} is {}".format(sp, num / den))

        print("\n")

        Qmax = np.max(snapshot_test[:, sp * Nt:(sp + 1) * Nt])
        Qmin = np.min(snapshot_test[:, sp * Nt:(sp + 1) * Nt])
        fig, axs = plt.subplots(1, 2, num=11, sharey=True, figsize=(10, 4))
        axs[0].pcolormesh(np.squeeze(qtilde), vmin=Qmin, vmax=Qmax)
        axs[0].set_ylabel(r'space $x$')
        axs[0].set_xlabel(r'time $t$')
        axs[0].set_title(r"NN")

        axs[1].pcolormesh(snapshot_test[:, sp * Nt:(sp + 1) * Nt], vmin=Qmin, vmax=Qmax)
        axs[1].set_ylabel(r'$i=1,\dots,M$')
        axs[1].set_xlabel(r'time $t$')
        axs[1].set_title(r"Data")
        # plt.tight_layout()
        # plt.show()
