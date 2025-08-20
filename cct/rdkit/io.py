from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from rdkit import Chem

from cct.rdkit.conformers import has_conformer
from cct.rdkit.connectivity import determine_bonds
from cct.rdkit.coordinates import set_coordinates, set_zero_coordinates


def rdkit_from_ase(atoms, charge: int = 0):
    mol = Chem.RWMol()
    atom_indices = []

    for atom in atoms:
        rd_atom = Chem.Atom(int(atom.number))
        idx = mol.AddAtom(rd_atom)
        atom_indices.append(idx)

    mol = mol.GetMol()

    set_zero_coordinates(mol)
    set_coordinates(mol, atoms.get_positions(), conf_id=0)
    determine_bonds(mol, charge=charge)
    return mol


def ase_from_rdkit(mol: Chem.Mol, conf_id: int = 0) -> Atoms:
    """Convert RDKit molecule to ASE Atoms object."""

    if not has_conformer(mol, conf_id=conf_id):
        raise ValueError("Mol does not have conformer")

    conf = mol.GetConformer(int(conf_id))
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
    atoms = Atoms(symbols=symbols, positions=positions)
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    # Store total charge in atoms.info
    atoms.info["charge"] = total_charge
    return atoms


def xyz_from_rdkit(mol: Chem.Mol, path: Path | str) -> None:
    """Dump *all* conformers of `mol` to an XYZ file at `path`."""
    path = Path(path)

    all_atoms = []
    for conf_id in range(mol.GetNumConformers()):
        atoms = ase_from_rdkit(mol, conf_id=conf_id)
        conf = mol.GetConformer(conf_id)
        if conf.HasProp("energy"):
            atoms.info["energy"] = float(conf.GetProp("energy"))

        all_atoms.append(atoms)

    write(path, images=all_atoms)


def rdkit_from_xyz(filename: Path, ref_mol: Chem.Mol | None = None, charge: int | None = None) -> Chem.Mol:
    """Read multi-molecule XYZ using ASE and add as conformers to template molecule.

    Args:
    ----
        filename: XYZ file with multiple structures
        ref_mol: RDKit molecule with correct connectivity

    Returns:
    -------
        mol: RDKit molecule with all conformers added
        energies: List of energies (one per conformer), extracted from comment lines

    """
    if ref_mol is None and charge is None:
        raise ValueError("Must provide either reference molecule or explicitly provide charge of the molecule")
    all_atoms = read(filename, index=":")

    if not ref_mol:
        ref_mol = rdkit_from_ase(all_atoms[0], charge=charge)

    mol = Chem.Mol(ref_mol)
    mol.RemoveAllConformers()

    for conf_id, atoms in enumerate(all_atoms):
        positions = atoms.get_positions()

        if "energy" in atoms.info:
            energy = atoms.info["energy"]
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.SetId(conf_id)
            conf.SetProp("energy", str(energy))

        set_coordinates(mol, coords=positions, conf_id=conf_id)

    return mol
