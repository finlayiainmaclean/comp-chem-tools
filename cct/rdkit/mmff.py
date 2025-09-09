import logging

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


def mmff_optimise(mol: Chem.Mol, constrained_atom_idxs: list[int] | None = None) -> tuple[float, Chem.Mol]:
    """Optimize molecular geometry using MMFF force field."""
    # Define a force field with constraints on non-hydrogen atoms
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94"))

    if constrained_atom_idxs is not None:
        for atom in mol.GetAtoms():
            if atom.GetIdx() in constrained_atom_idxs:
                ff.AddFixedPoint(atom.GetIdx())
    try:
        ff.Minimize()
    except Exception:
        logger.warning(f"Failed to minimize molecule {Chem.MolToSmiles(mol)}")
    # Get the optimized structure and energy
    energy = ff.CalcEnergy()
    return mol, energy
