import numpy as np
from ase import Atoms
from rdkit import Chem

from cct.rdkit.conformers import set_coordinates


def rdkit_to_ase(mol: Chem.Mol) -> Atoms:
    """Convert RDKit molecule to ASE Atoms object."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Mol has no 3-D conformer; generate coordinates first.")

    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = np.array(
        [
            (
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            )
            for i in range(mol.GetNumAtoms())
        ],
        dtype=float,
    )
    return Atoms(symbols=symbols, positions=positions)


def ase_to_rdkit(atoms: Atoms, rdmol: Chem.Mol) -> Chem.Mol:
    """Convert ASE Atoms object to RDKit molecule."""
    set_coordinates(rdmol, atoms.get_positions())
    return rdmol
