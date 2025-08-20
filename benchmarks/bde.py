import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from cct.ase_calcs import ase_optimise, ase_singlepoint
from cct.rdkit.conformers import embed
from cct.rdkit.io import ase_from_rdkit


def get_bonds_from_atom(mol, atom_idx):
    """Get all bonds where the specified atom index is involved"""
    bonds = []
    atom = mol.GetAtomWithIdx(atom_idx)

    for bond in atom.GetBonds():
        bonds.append(bond)

    return bonds


def sort_mols_by_size(mols):
    """Sort fragments by number of atoms (ascending)"""
    return sorted(mols, key=lambda mol: mol.GetNumAtoms())


def func(smi, atom_idx, method="gxTB"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    mol = embed(mol)
    monomer_atoms = ase_from_rdkit(mol, conf_id=0)
    monomer_atoms, _ = ase_optimise(monomer_atoms, multiplicity=1, charge=0, calc="GFN2-xTB", fmax=0.05)
    # monomer_atoms, _ = ase_optimise(monomer_atoms, multiplicity=1, charge=0, method=method, fmax=0.01)

    E_monomer = ase_singlepoint(monomer_atoms, multiplicity=1, charge=0, calc=method)

    fragB_atoms = monomer_atoms[[atom_idx]].copy()
    fragA_atoms = monomer_atoms.copy()
    del fragA_atoms[atom_idx]

    if len(fragA_atoms) > 1:
        fragA_atoms, _ = ase_optimise(fragA_atoms, multiplicity=2, charge=0, calc="GFN2-xTB", fmax=0.05)
        # fragA_atoms, _ = ase_optimise(fragA_atoms, multiplicity=2, charge=0, method=method, fmax=0.01)

    E_fragA = ase_singlepoint(fragA_atoms, multiplicity=2, charge=0, calc=method)

    E_fragB = ase_singlepoint(fragB_atoms, multiplicity=2, charge=0, calc=method)

    BDE_eV = (E_fragA + E_fragB) - E_monomer
    BDE_kcal_mol = BDE_eV * 23.061
    return BDE_kcal_mol


if __name__ == "__main__":
    qm_method = "UMA_SMALL"  # "gxTB"
    df = pd.read_csv("data/supporting-information.csv")
    df = df[df.columns[:6].tolist() + ["eSEN-S", "GFN2-xTB"]]

    df = df[~df.SMILES.str.contains("Si")]
    exp_col = "Reference BDE"
    raw_pred_col = "eBDE"
    corrected_pred_col = f"{raw_pred_col}_corrected"
    df[raw_pred_col] = [
        func(smi=row.SMILES, atom_idx=row["Atom index"] - 1, method=qm_method)
        for i, row in tqdm(df.iterrows(), total=len(df))
    ]
    reg = LinearRegression()
    reg.fit(df[[raw_pred_col]].values, df[exp_col])
    df[corrected_pred_col] = reg.predict(df[[raw_pred_col]].values)

    mae = mean_absolute_error(df[exp_col], df[corrected_pred_col])
    rmse = np.sqrt(mean_squared_error(df[exp_col], df[corrected_pred_col]))
    r2 = r2_score(df[exp_col], df[corrected_pred_col])
    tau, _ = kendalltau(df[exp_col], df[corrected_pred_col])
    print(f"r2: {r2:.2f}")
    print(f"mae: {mae:.2f}")
    print(f"rmse: {rmse:.2f}")
    print(f"w: {reg.coef_[0]:.2f}")
    print(f"b: {reg.intercept_:.2f}")
