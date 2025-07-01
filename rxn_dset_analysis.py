"""
Analysis for rxn dataset with varying noise levels for different combinations of known
concentrations.

This module provides functions to perform NMF on rxn dataset with varying noise levels
using FroALS, FroFPGM, and MinVol algorithms. It includes parallel processing
capabilities to analyze multiple combinations of known concentration points efficiently.
"""

import itertools
import os
import os.path as osp
import multiprocessing


from joblib import Parallel, delayed
import numpy as np
from mcrnmf import MinVol, FroFPGM, FroALS, SNPA
import pandas as pd

RXN_DATA_DPATH = osp.join(os.getcwd(), "datasets", "rxn")


def get_W_truth_data():
    """
    Load the ground truth pure component spectra.

    Returns
    -------
    intensity : numpy.ndarray
        Pure component spectra intensity values.
    wavenumber : numpy.ndarray
        Wavenumber values corresponding to the intensity data.
    """
    fname = "pure_spectra.csv"
    fpath = osp.join(RXN_DATA_DPATH, fname)
    data_arr = np.loadtxt(fpath, delimiter=",", skiprows=1)
    intensity = data_arr[:, 1:]
    wavenumber = data_arr[:, 0]

    return intensity, wavenumber


def get_X_data(dataset_name: str, noise_perc: int = 0):
    """
    Load the spectroscopic data matrix for a given dataset and noise level.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load (e.g., 'rxn1', 'rxn2').
    noise_perc : int, optional
        Percentage of noise in the data, default is 0.

    Returns
    -------
    numpy.ndarray
        Spectroscopic data matrix X with dimensions (wavelengths, time points).
    """
    # load the X data
    fname_X = f"X_100dPt_{dataset_name}_{noise_perc}PercNoise.csv"
    X = np.loadtxt(osp.join(RXN_DATA_DPATH, fname_X), delimiter=",", dtype=np.float64)
    return X


def get_H_truth_data(dataset_name: str):
    """
    Load the ground truth concentration profiles for a given dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load (e.g., 'rxn1', 'rxn2').

    Returns
    -------
    numpy.ndarray
        Ground truth concentration profiles with dimensions (components, time points).
    """
    # load the H data
    fname_H = f"H_100dPt_{dataset_name}.csv"
    H = np.loadtxt(osp.join(RXN_DATA_DPATH, fname_H), delimiter=",", dtype=np.float64)

    return H


def gen_known_H(H_truth: np.ndarray, combination: tuple[int]):
    """
    Generate a partially known concentration matrix for the specified combination of time points.

    Creates a matrix with NaN values except at the specified time points (combination)
    and the first and last time points, which are always included.

    Parameters
    ----------
    H_truth : numpy.ndarray
        Ground truth concentration profiles.
    combination : tuple[int]
        Tuple of time point indices to include as known values.

    Returns
    -------
    numpy.ndarray
        Partially known concentration matrix with NaN values at unknown positions.
    """
    known_H = np.full_like(H_truth, fill_value=np.nan)
    # add the first and last time points because it assumed to be always known
    idx_ls = [0] + list(combination) + [-1]
    known_H[:, idx_ls] = H_truth[:, idx_ls]

    return known_H


def nmf_permute_W_and_H(W: np.ndarray, H: np.ndarray, W_truth: np.ndarray):
    """
    Rearrange the estimated W columns to match the order of W_truth.

    Since NMF solutions are invariant to permutations of the components,
    this function reorders the components to facilitate comparison with ground truth.

    Parameters
    ----------
    W : numpy.ndarray
        Estimated spectra matrix.
    H : numpy.ndarray
        Estimated concentration matrix.
    W_truth : numpy.ndarray
        Ground truth spectra matrix.

    Returns
    -------
    W_rearranged : numpy.ndarray
        Rearranged spectra matrix with columns matching W_truth order.
    H_rearranged : numpy.ndarray
        Rearranged concentration matrix with rows matching the new W order.
    """
    # shape of diff (#wavelengths, ncol_W, n_pure)
    diff = W[:, :, np.newaxis] - W_truth[:, np.newaxis, :]
    # shape of norms is (ncol_W, n_pure)
    norms = np.linalg.norm(diff, axis=0)
    best_match_indices = np.argmin(norms, axis=0)
    W_rearranged = W[:, best_match_indices].copy()
    H_rearranged = H[best_match_indices, :].copy()

    return W_rearranged, H_rearranged


def process_known_H(
    X: np.ndarray,
    Wi: np.ndarray,
    Hi: np.ndarray,
    W_truth: np.ndarray,
    H_truth: np.ndarray,
    norm_W_truth: float,
    norm_H_truth: float,
    combination: tuple[int],
    model_name: str,
    rank: int,
    constraint_kind: int,
    unimodal_constraints: dict[str, bool],
    iter_max: int,
    tol: float,
    noise_perc: int,
    dataset: str,
    lambdaa: float | None = None,
):
    """
    Perform NMF with partially known concentrations using the specified model.

    This function runs either MinVol, FroFPGM, or FroALS for a specific combination
    of known concentration values and calculates error metrics.

    Parameters
    ----------
    X : numpy.ndarray
        Spectroscopic data matrix.
    Wi : numpy.ndarray
        Initial guess for the spectra matrix.
    Hi : numpy.ndarray
        Initial guess for the concentration matrix.
    W_truth : numpy.ndarray
        Ground truth spectra matrix.
    H_truth : numpy.ndarray
        Ground truth concentration matrix.
    norm_W_truth : float
        Frobenius norm of the ground truth spectra matrix.
    norm_H_truth : float
        Frobenius norm of the ground truth concentration matrix.
    combination : tuple[int]
        Tuple of time point indices to include as known values.
    model_name : str
        Name of the NMF model to use ('mvol', 'fpgm', or 'als').
    rank : int
        Number of components to factorize into.
    constraint_kind : int
        Type of constraints to apply (see mcrnmf documentation).
    unimodal_constraints : dict[str, bool]
        Whether to apply unimodality constraints.
    iter_max : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance.
    noise_perc : int
        Percentage of noise in the data.
    dataset : str
        Name of the dataset.
    lambdaa : float or None, optional
        Regularization parameter for MinVol, default is None.

    Returns
    -------
    dict
        Dictionary containing results and error metrics.
    """
    if model_name not in ["mvol", "fpgm", "als"]:
        raise ValueError("`model_name` must be either 'mvol', 'fpgm', or 'als'")
    model: MinVol | FroALS | FroFPGM
    if model_name == "mvol":
        if lambdaa is None:
            raise ValueError("`lambdaa` cannot be None, while trying to run MinVol")
        model = MinVol(
            rank=rank,
            constraint_kind=constraint_kind,
            unimodal=unimodal_constraints,
            iter_max=iter_max,
            tol=tol,
            lambdaa=lambdaa,
        )
    elif model_name == "fpgm":
        model = FroFPGM(
            rank=rank,
            constraint_kind=constraint_kind,
            unimodal=unimodal_constraints,
            iter_max=iter_max,
            tol=tol,
        )
    elif model_name == "als":
        model = FroALS(
            rank=rank,
            constraint_kind=constraint_kind,
            unimodal=unimodal_constraints,
            iter_max=iter_max,
            tol=tol,
        )

    known_H = gen_known_H(H_truth, combination)
    model.fit(X, Wi, Hi, known_H=known_H)
    W, H = nmf_permute_W_and_H(model.W, model.H, W_truth)

    res_dict = {
        "points": combination,
        "converged": model.is_converged,
        "% Rel-MSE X": model.rel_reconstruction_error_ls[-1] * 100,
        "% Rel-MSE W": np.linalg.norm(W - W_truth, ord="fro") * 100 / norm_W_truth,
        "% Rel-MSE H": np.linalg.norm(H - H_truth, ord="fro") * 100 / norm_H_truth,
        # -1 because 1st element is the error before 1st iteration.
        "iter_perf": len(model.rel_reconstruction_error_ls) - 1,
        "tol": tol,
        "unimodal": unimodal_constraints["H"],
        "iter_max": iter_max,
        "constraint_kind": constraint_kind,
        "model": model_name,
        "noise [%]": noise_perc,
        "dataset": dataset,
        "relative loss": model.rel_loss_ls[-1],
        "lambda": lambdaa if model_name == "mvol" else np.nan,
    }

    return res_dict


def run_all_comb_known_H(
    results_save_dpath: str,
    num_threads: int,
    model_name: str,
    dataset_name: str,
    noise_perc: int,
    unimodal_constraints: dict[str, bool],
    tolerance: float,
    lambdaa: float | None = None,
    num_combinations: int | None = None,
):
    """
    Run NMF for all combinations of two additional known concentration points.

    This function performs NMF using the specified model for all possible combinations
    of two additional known time points (in addition to the first and last points),
    and saves the results to a CSV file.

    Parameters
    ----------
    results_save_dpath : str
        Path to save the results.
    num_threads : int
        Number of threads to use for parallel processing.
    model_name : str
        Name of the NMF model to use ('mvol', 'fpgm', or 'als').
    dataset_name : str
        Name of the dataset to analyze.
    noise_perc : int
        Percentage of noise in the data.
    unimodal_constraints : dict[str, bool]
        Whether to apply unimodality constraints.
    tolerance : float
        Convergence tolerance.
    lambdaa : float or None, optional
        Regularization parameter for MinVol, default is None.
    num_combinations : int or None, optional
        Number of combinations to process (for testing), default is None (all combinations).
    """
    if (model_name == "mvol") and (lambdaa is None):
        raise ValueError("`lambdaa` cannot be None if the `model_name` is 'mvol'")

    W_truth, _ = get_W_truth_data()
    X = get_X_data(dataset_name, noise_perc=noise_perc)
    H_truth = get_H_truth_data(dataset_name)

    # generate all combinations of the two additional known_H points
    num_pts = X.shape[1]
    comb_ls = list(itertools.combinations(range(1, num_pts - 1), 2))
    if num_combinations is not None:
        comb_ls = comb_ls[:num_combinations]

    # parameters for model
    rank = 3
    constraint_kind = 4
    unimodal_constraints = unimodal_constraints
    iter_max = 2000
    tol = tolerance
    lambdaa_val = lambdaa
    # generate initial guess
    snpa = SNPA(rank=3)
    snpa.fit(X)
    sort_order = np.argsort(snpa.col_indices_ls)  # sort to have the correct order
    Wi = np.ascontiguousarray(snpa.W[:, sort_order], dtype=np.float64)
    Hi = np.ascontiguousarray(snpa.H[sort_order, :], dtype=np.float64)
    norm_W_truth = np.linalg.norm(W_truth, ord="fro")
    norm_H_truth = np.linalg.norm(H_truth, ord="fro")

    # the parallel processing job
    results = Parallel(n_jobs=num_threads, verbose=2)(
        delayed(process_known_H)(
            X,
            Wi,
            Hi,
            W_truth,
            H_truth,
            norm_W_truth,
            norm_H_truth,
            combination,
            model_name,
            rank,
            constraint_kind,
            unimodal_constraints,
            iter_max,
            tol,
            noise_perc,
            dataset_name,
            lambdaa_val,
        )
        for combination in comb_ls
    )

    # convert results to dataframe
    df = pd.DataFrame(results)

    if osp.exists(results_save_dpath) is False:
        os.makedirs(results_save_dpath)
    save_fname = (
        f"{model_name}_{dataset_name}_{noise_perc}noisePerc_tol_{tol:.0e}_unimodalH_"
        f"{str(unimodal_constraints["H"])}"
    )
    if model_name == "mvol":
        save_fname = save_fname + f"_lambda_{lambdaa:.0e}"

    df.to_csv(osp.join(results_save_dpath, save_fname + ".csv"), index=False)


def validate_thread_count(requested_threads: int) -> int:
    """
    Validates that the requested number of threads doesn't exceed system capacity
    """
    # get available CPU count
    available_threads = multiprocessing.cpu_count()

    if requested_threads > available_threads:
        supplied_threads = round(available_threads * 0.75)
        print(
            f"\nWarning: Requested {requested_threads} threads, but only "
            f"{available_threads} are available."
        )
        print(
            f"\nAdjusting to use {supplied_threads} of {available_threads} threads "
            f"instead.\n"
        )
        return supplied_threads
    else:
        return requested_threads


if __name__ == "__main__":
    # the value of 10 is used because my laptop has 14 cores. Adjust this value according.
    num_threads = 10
    num_threads = validate_thread_count(num_threads)
    unimodal_constraints = {"H": True}
    tolerance = 1e-4
    datasets = ["rxn1", "rxn2"]
    noise_perc_ls = [0, 2, 5]
    results_save_dpath = osp.join(os.getcwd(), "rxn-comb-results")

    # inform the user the files in the specified path will be overwritten
    message = (
        f"Running the code will overwrite the current results files (if any) in the path"
        f"\n{results_save_dpath}.\nPlease confirm you want to proceed (yes/no):  "
    )
    while True:
        response = input(message).strip().lower()
        if response in ["yes", "y"]:
            # Perform ALS and FPGM
            model_ls = ["als", "fpgm"]
            for model in model_ls:
                for dataset in datasets:
                    for noise_perc in noise_perc_ls:
                        print(
                            f"Performing {model} NMF on {dataset} dataset with {noise_perc} % "
                            "noise "
                        )
                        print(90 * "+")
                        run_all_comb_known_H(
                            results_save_dpath=results_save_dpath,
                            num_threads=num_threads,
                            model_name=model,
                            dataset_name=dataset,
                            noise_perc=noise_perc,
                            unimodal_constraints=unimodal_constraints,
                            tolerance=tolerance,
                        )
                        print(90 * "+")
                        print("")

            # Perform MinVol
            lambdaa_ls = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
            model = "mvol"
            for dataset in datasets:
                for noise_perc in noise_perc_ls:
                    for lambdaa in lambdaa_ls:
                        print(
                            f"Performing {model} NMF on {dataset} with {noise_perc} % noise "
                            f"using lambda {lambdaa:.0e}"
                        )
                        print(90 * "+")
                        run_all_comb_known_H(
                            results_save_dpath=results_save_dpath,
                            num_threads=num_threads,
                            model_name="mvol",
                            dataset_name=dataset,
                            noise_perc=noise_perc,
                            unimodal_constraints=unimodal_constraints,
                            tolerance=tolerance,
                            lambdaa=lambdaa,
                        )
                        print(90 * "+")
                        print("")
            break
        elif response in ["no", "n"]:
            print("Execution cancelled.")
            break
        else:
            print("Please answer with yes or no")
