import os
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

from cct.ase_calcs import CalculatorFactory
from cct.consts import EV2KCALMOL
from cct.rdkit.conformers import keep_conformers, remove_high_energy_conformers, sort_conformers_by_values
from cct.utils import batch_run, boltzmann_average_energy, setup_logger
from cct.workflows.conformers import _generate_conformers_rdkit

logger = setup_logger(__name__)
calcs = CalculatorFactory()
enumerator = rdMolStandardize.TautomerEnumerator()
num_cores = cpu_count()
RUN_LOCAL = eval(os.environ.get("RUN_LOCAL", "False"))


max_conformers_qm = 5
energy_window = 5 / EV2KCALMOL
qm_method = "eSEN-S"
solv_method = "GFN2-xTB"
calculate_thermo = False


def process_both(smi1, smi2):
    e1 = process_one(smi1)
    e2 = process_one(smi2)
    ddg = (e2 - e1) * EV2KCALMOL
    return ddg


def process_one(smi, multiplicity: int = 1):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    charge = Chem.GetFormalCharge(mol)
    mol = _generate_conformers_rdkit(mol, mode="rapid", multiplicity=multiplicity, charge=charge, solvent="water")

    # Single point calculations
    print(f"Running QM singlepoint with {mol.GetNumConformers()} conformers")

    energies = calcs.singlepoint_mol(mol, method=qm_method, multiplicity=multiplicity, charge=charge)

    mol, energies = remove_high_energy_conformers(mol, energies=energies, energy_window=energy_window)
    conf_ids = np.argsort(energies)[:max_conformers_qm]
    mol = keep_conformers(mol, conf_ids=conf_ids)

    print(f"Running QM optimisation with {mol.GetNumConformers()} conformers")

    # Optimization
    mol, gas_phase_energies = calcs.optimise_mol(
        mol, method=qm_method, multiplicity=multiplicity, charge=charge, fmax=0.01
    )

    # Combined energies
    if calculate_thermo:
        gas_phase_energies = []
        for conf_id in range(mol.GetNumConformers()):
            g, _ = calcs.thermochemistry_conf(mol, conf_id=conf_id, method=qm_method)
            gas_phase_energies.append(g)
        gas_phase_energies = np.array(gas_phase_energies)

    # Solvation energy
    solvation_energies = calcs.solvation_energy_mol(mol, multiplicity=multiplicity, charge=charge, method=solv_method)

    solvated_energies = gas_phase_energies + solvation_energies
    mol, solvated_energies = sort_conformers_by_values(mol, values=solvated_energies)

    avg_solvated_energy = boltzmann_average_energy(solvated_energies)
    return avg_solvated_energy


def generate_plot(df, pred_col="ddg_pred", exp_col="ddg", energy_range=(-3, 3), figsize=(8, 8)):
    """
    Create a parity plot highlighting data within a specific energy range.
    Red points are within the energy range, black points are all data.
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Create energy range filter
    energy_mask = (df[exp_col] >= energy_range[0]) & (df[exp_col] <= energy_range[1])

    # Plot all data in black first (so red points appear on top)
    ax.scatter(df[pred_col], df[exp_col], c="black", alpha=0.6, s=25, label="All data")

    # Plot energy range subset in red on top
    subset_df = df[energy_mask]
    ax.scatter(
        subset_df[pred_col],
        subset_df[exp_col],
        c="red",
        alpha=0.8,
        s=25,
        label=f"Within {energy_range[0]} to {energy_range[1]} kcal/mol",
    )

    # Perfect prediction line
    lims = [-20, 20]
    ax.plot(lims, lims, "--", color="gray", alpha=0.8, linewidth=2, zorder=0)

    # Calculate regression metrics for ALL data
    mae = mean_absolute_error(df[exp_col], df[pred_col])
    rmse = np.sqrt(mean_squared_error(df[exp_col], df[pred_col]))
    r2 = r2_score(df[exp_col], df[pred_col])
    tau, _ = kendalltau(df[exp_col], df[pred_col])
    n = len(df)

    # Calculate binary classification accuracies
    # All data accuracy
    y_true = df[exp_col].apply(lambda x: 0 if x < 0 else 1)
    y_pred = df[pred_col].apply(lambda x: 0 if x < 0 else 1)
    acc_all = accuracy_score(y_true, y_pred)

    # Red subset accuracy (within energy range)
    df_red = df[df[exp_col].abs() < 3]
    y_true_red = df_red[exp_col].apply(lambda x: 0 if x < 0 else 1)
    y_pred_red = df_red[pred_col].apply(lambda x: 0 if x < 0 else 1)
    acc_red = accuracy_score(y_true_red, y_pred_red)

    # Add metrics text box with accuracies
    metrics_text = (
        f"MAE    {mae:.2f}\n"
        f"RMSE  {rmse:.2f}\n"
        f"r²       {r2:.2f}\n"
        f"τ        {tau:.2f}\n"
        f"N        {n}\n"
        f"Acc      {acc_all:.2f}\n"
        f"Acc₃     {acc_red:.2f}"
    )

    ax.text(
        0.05,
        0.95,
        metrics_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Styling
    ax.set_xlabel("Predicted ΔΔG", fontsize=14)
    ax.set_ylabel("Experimental ΔΔG", fontsize=14)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    df = pd.read_csv(
        "https://raw.githubusercontent.com/choderalab/tautomer-data/refs/heads/main/data/input/b3lyp_tautobase_subset.txt"
    )
    df.columns = ["name", "t1", "t2", "ddg"]

    df["l1"] = df.t1.apply(len)
    df = df.sort_values(by="l1", ascending=False)
    df = df.head(20)

    runtime_env = {
        "num_cpus": 1,
        "runtime_env": {
            "env_vars": {
                "RAY_DISABLE_MEMORY_MONITOR": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            }
        },
    }

    df["ddg_pred"] = batch_run(
        process_both, list(zip(df.t1.tolist(), df.t2.tolist())), ray_kwargs=runtime_env, run_local=RUN_LOCAL
    )
    reg = LinearRegression()
    reg.fit(df[["ddg_pred"]].values, df.ddg.values)
    df["ddg_pred_corrected"] = reg.predict(df[["ddg_pred"]].values)
    df.to_csv("tautbase.csv")

    # df['ddg_pred_corrected'] = 0.544689*df['ddg_pred']-1.00448

    pd.Series([reg.coef_[0], reg.intercept_]).to_csv("tautbase_weights.csv", index=False, header=None)

    fig, ax = generate_plot(df, pred_col="ddg_pred", exp_col="ddg")
    plt.savefig("tautbase_raw.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = generate_plot(df, pred_col="ddg_pred_corrected", exp_col="ddg")
    plt.savefig("tautbase_corrected.png", dpi=300, bbox_inches="tight")
    plt.close()
