"""
Generates Rxn dataset

This module creates synthetic spectroscopic data of Rxn1 and Rxn2 datasets. It generates
concentration profiles using kinetic models and combines them with pure component spectra
to create spectroscopic data matrices, optionally with added noise.
"""

import os
import os.path as osp

import numpy as np
import pandas as pd
from scipy.linalg import expm


def gen_rxn_data(num_time_pts=100, noise_frac=None):
    """
    Generate synthetic reaction spectroscopic data with optional noise.

    This function creates synthetic spectroscopic data for two different reaction
    systems by combining pure component spectra with concentration profiles generated
    from kinetic models. The data is saved as CSV files in the 'datasets/rxn' directory.

    Parameters
    ----------
    num_time_pts : int, optional
        Number of time points to generate in the concentration profiles, default is 100.
    noise_frac : float or None, optional
        Fraction of maximum signal to use for noise standard deviation.
        If None, no noise is added. Otherwise, both intensity noise and
        baseline noise are added to the spectroscopic data.

    Notes
    -----
    The function generates the following files for each reaction system:
    - H_<num_time_pts>dPt_rxn<i>.csv: Concentration profiles
    - X_<num_time_pts>dPt_rxn<i>_<noise>PercNoise.csv: Spectroscopic data

    The two reaction systems have different rate constant matrices:
    - rxn1: K = [[-0.8, 0, 0], [0.8, -0.2, 0], [0, 0.2, 0]]
    - rxn2: K = [[-0.2, 0, 0], [0.2, -0.8, 0], [0, 0.8, 0]]

    Both systems start with only the first component present (c_initial = [1, 0, 0]).
    """
    cwd = os.getcwd()
    rxn_dpath = osp.join(cwd, "datasets", "rxn")

    # get the pure spectra
    pure_spectra_df = pd.read_csv(osp.join(rxn_dpath, "pure_spectra.csv"))
    pure_spectra = pure_spectra_df.iloc[:, 1:].to_numpy(dtype=np.float64)

    K_ls = []
    # rxn1
    K_ls.append(np.array([[-0.8, 0, 0], [0.8, -0.2, 0], [0, 0.2, 0]]))
    # rxn2
    K_ls.append(np.array([[-0.2, 0, 0], [0.2, -0.8, 0], [0, 0.8, 0]]))

    # number of time points
    t = np.linspace(0, 50, num=num_time_pts)

    for i in range(len(K_ls)):
        K = K_ls[i]
        c_initial = np.zeros(K.shape[0])
        c_initial[0] = 1
        C = np.zeros((K_ls[0].shape[0], t.size))
        for j in range(t.size):
            C[:, j] = np.dot(expm(K * t[j]), c_initial)
        C = C / C.sum(axis=0)
        X = pure_spectra @ C
        H_fname = f"H_{t.size}dPt_rxn{i+1}.csv"
        if noise_frac is None:
            X_fname = f"X_{t.size}dPt_rxn{i+1}_{0}PercNoise.csv"
        else:
            # set random number generators for reproducibility
            rng_intensity = np.random.default_rng(2)
            rng_baseline = np.random.default_rng(50)
            baseline_noise = np.abs(rng_baseline.normal(0, 0.001, X.shape))
            intensity_noise = rng_intensity.normal(0, noise_frac * X.max(), X.shape)
            X = X + np.abs(intensity_noise) + baseline_noise
            assert np.all(X) >= 0
            noise_perc = noise_frac * 100
            if noise_perc < 1:
                X_fname = f"X_{t.size}dPt_rxn{i+1}_{noise_perc:.1f}PercNoise.csv"
            else:
                noise_perc = int(noise_perc)
                X_fname = f"X_{t.size}dPt_rxn{i+1}_{noise_perc}PercNoise.csv"

        pd.DataFrame(data=X).to_csv(
            osp.join(rxn_dpath, X_fname), header=None, index=None
        )
        pd.DataFrame(data=C).to_csv(
            osp.join(rxn_dpath, H_fname), header=None, index=None
        )


if __name__ == "__main__":
    """
    Generate reaction data with noise percent of [0, 2, 5] when script is run directly.
    """
    for noise_frac in [None, 0.02, 0.05]:
        gen_rxn_data(num_time_pts=100, noise_frac=noise_frac)
