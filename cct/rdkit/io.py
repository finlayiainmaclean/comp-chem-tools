from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from rdkit import Chem

from cct.rdkit.coordinates import set_coordinates


def to_ase(mol: Chem.Mol, conf_id: int = -1) -> Atoms:
    """Convert RDKit molecule to ASE Atoms object."""
    if mol.GetNumConformers() == 0:
        raise ValueError("Mol has no 3-D conformer; generate coordinates first.")

    conf = mol.GetConformer(conf_id)
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


def to_xyz(mol: Chem.Mol, path: Path | str) -> None:
    """Dump *all* conformers of `mol` to an XYZ file at `path`."""
    path = Path(path)

    mol_blocks = []
    for cid in range(mol.GetNumConformers()):
        conf = mol.GetConformer(cid)

        # Get the standard XYZ block
        xyz_block = Chem.MolToXYZBlock(mol, confId=cid)
        lines = xyz_block.strip().split("\n")

        val = ""
        if conf.HasProp("Energy"):
            val = conf.GetProp("Energy")

        lines[1] = val
        mol_blocks.append("\n".join(lines) + "\n")

    mol_block = "".join(mol_blocks)
    path.write_text(mol_block)


def from_xyz(filename: Path, ref_mol: Chem.Mol) -> Chem.Mol:
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
    all_atoms = read(filename, index=":")

    mol = Chem.Mol(ref_mol)
    mol.RemoveAllConformers()

    for conf_id, atoms in enumerate(all_atoms):
        positions = atoms.get_positions()

        try:
            comment_info = list(atoms.info.keys())  # atoms.info = {'-5.55838': True}
            energy = float(comment_info[0])
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.SetId(conf_id)
            conf.SetProp("Energy", str(energy))
        except Exception:
            print(f"{conf_id} doesn't have energy in xyz comment line")

        mol.AddConformer(conf, assignId=True)

        set_coordinates(mol, positions, conf_id=conf_id)

    return mol
