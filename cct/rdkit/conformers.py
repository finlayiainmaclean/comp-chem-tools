import copy
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.spatial.distance import cdist

from cct.rdkit.mmff import mmff_optimise

logger = logging.getLogger(__name__)


def get_coordinates(mol: Chem.Mol, conf_id: int | None = 0) -> np.ndarray:
    """Get coordinates from molecular conformer."""
    coords = mol.GetConformer(conf_id).GetPositions()
    return coords


def set_random_coordinates(mol: Chem.Mol):
    """Set random coordinates for a molecule using ETKDG."""
    ps = Chem.AllChem.ETKDGv3()
    ps.useRandomCoords = True
    AllChem.EmbedChem.Mol(mol, ps)


def set_coordinates(mol: Chem.Mol, coords: np.ndarray, conf_id: int = -1):
    """Overwrite (or create) conformer with supplied Cartesian coordinates.

    Overwrites (or creates) conformer `conf_id` with the supplied
    Cartesian coordinates (Å).  `coords` must be shape (N_atoms, 3).
    """
    n_atoms = mol.GetNumAtoms()
    if coords.shape != (n_atoms, 3):
        raise ValueError(
            f"coords shape {coords.shape} does not match atom count {n_atoms}"
        )

    # make sure the conformer exists
    if mol.GetNumConformers() == 0:
        set_random_coordinates(mol)

    conf = mol.GetConformer(conf_id)

    # write xyz into the conformer
    for idx, (x, y, z) in enumerate(coords.astype(float)):
        conf.SetAtomPosition(idx, Point3D(x, y, z))


def extract_conformer(mol: Chem.Mol, /, cid: int) -> Chem.Mol:
    """Extract a single conformer from a multi-conformer molecule."""
    single_conf = Chem.Mol(mol)
    single_conf.RemoveAllConformers()
    single_conf.AddConformer(mol.GetConformer(cid), assignId=True)
    return single_conf


def transplant_coordinates(ref: Chem.Mol, query: Chem.Mol) -> Chem.Mol:
    """Transplant coordinates from reference molecule to query molecule."""
    DISTANCE_THRESHOLD = 0.25

    query_noh = Chem.RemoveHs(query)
    ref_noh = Chem.RemoveHs(ref)

    query_copy = copy.deepcopy(query_noh)
    ref_copy = copy.deepcopy(ref_noh)

    assert ref_noh.GetNumAtoms() == query_noh.GetNumAtoms()

    for atom in query_copy.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)

    for atom in ref_copy.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)

    match = ref_copy.GetSubstructMatch(query_copy)
    assert len(match) == ref_noh.GetNumAtoms()
    coords = get_coordinates(ref_noh)
    set_coordinates(query_noh, coords[np.array(match)])  # Set coords of heavy atoms
    query = Chem.AddHs(query_noh, addCoords=True)  # Add any missing hydrogens

    query_coords = get_coordinates(query)  # with explicit hydrogens
    ref_coords = get_coordinates(ref)  # with explicit hydrogens

    dist = cdist(query_coords, ref_coords)
    q_ix, r_ix = zip(*np.argwhere(dist < DISTANCE_THRESHOLD).astype(int), strict=False)
    q_ix = np.array(q_ix)
    r_ix = np.array(r_ix)

    query_coords[q_ix] = ref_coords[
        r_ix
    ]  # Replace coords of any atom under distance threshold

    set_coordinates(query, query_coords)

    query, _ = mmff_optimise(query, constrained_atom_idxs=q_ix)
    return query
