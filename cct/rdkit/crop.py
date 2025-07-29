import copy
from functools import reduce

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import Bond
from scipy.spatial.distance import cdist

from cct.rdkit.conformers import get_coordinates


def is_bond_to_break(bond: Bond) -> bool:
    src = bond.GetBeginAtom()
    trg = bond.GetEndAtom()

    src_info = src.GetPDBResidueInfo()
    trg_info = trg.GetPDBResidueInfo()

    src_name = src_info.GetName().strip()
    trg_name = trg_info.GetName().strip()
    c_c_bond = (src_name == "C" and trg_name == "CA") or (
        src_name == "CA" and trg_name == "C"
    )
    s_s_bond = src_name == "SG" and trg_name == "SG"
    if (c_c_bond or s_s_bond) and bond.GetBondTypeAsDouble() == 1:
        return True
    else:
        return False


def crop_protein(
    protein_rdmol: Chem.Mol,
    ligand_rdmol: Chem.Mol,
    max_distance_to_ligand: float = 4,
):
    protein_rdmol = Chem.MolFromPDBBlock(
        Chem.MolToPDBBlock(protein_rdmol, flavor=4), removeHs=False
    )
    _ORIG_TAG = "_orig_idx"

    ligand_coords = get_coordinates(ligand_rdmol)
    protein_coords = get_coordinates(protein_rdmol)

    for atom in protein_rdmol.GetAtoms():
        atom.SetIntProp(_ORIG_TAG, atom.GetIdx())

    # Create copies so we don't modify the input Chem.Mols inplace
    # Make copy before we start breaking bonds
    final_protein_rdmol = Chem.EditableMol(protein_rdmol)
    protein_rdmol = Chem.EditableMol(protein_rdmol)
    ligand_rdmol = copy.deepcopy(ligand_rdmol)

    # First pass at breaking every possible single C-C or S-S bond
    for bond in protein_rdmol.GetMol().GetBonds():
        if is_bond_to_break(bond):
            protein_rdmol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    frag_membership = Chem.GetMolFrags(
        protein_rdmol.GetMol(), asMols=False, sanitizeFrags=False
    )

    # Build the part_list
    part_list = [None] * protein_rdmol.GetMol().GetNumAtoms()
    for comp_id, atom_idxs in enumerate(frag_membership):
        for idx in atom_idxs:
            part_list[idx] = comp_id

    atom_idx_to_group_idx = {
        idx: comp_id
        for comp_id, atom_idxs in enumerate(frag_membership)
        for idx in atom_idxs
    }

    min_dist_to_ligand = cdist(ligand_coords, protein_coords).min(axis=0)
    df = pd.DataFrame(zip(part_list, min_dist_to_ligand), columns=["group", "dist"])
    group_idx_under_threshold = (
        df.groupby("group")["dist"].min() < max_distance_to_ligand
    ).to_dict()

    # Second pass only breaking C-C or S-S bonds where their corresponding
    # mononer has no atoms under the threshold
    bonds_to_delete = []
    for bond in final_protein_rdmol.GetMol().GetBonds():
        src = bond.GetBeginAtom()
        trg = bond.GetEndAtom()

        keep_src = group_idx_under_threshold[atom_idx_to_group_idx[src.GetIdx()]]
        keep_trg = group_idx_under_threshold[atom_idx_to_group_idx[trg.GetIdx()]]
        keep_both = keep_src and keep_trg

        if is_bond_to_break(bond) and not keep_both:
            s, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bonds_to_delete.append((s, t))
            final_protein_rdmol.RemoveBond(s, t)
            src.SetNumExplicitHs(src.GetNumExplicitHs() + 1)
            trg.SetNumExplicitHs(trg.GetNumExplicitHs() + 1)

    # Add explcit hydrogens
    final_protein_rdmol = Chem.AddHs(
        final_protein_rdmol.GetMol(), explicitOnly=True, addCoords=True
    )

    protein_frag_rdmols = Chem.GetMolFrags(
        final_protein_rdmol, asMols=True, sanitizeFrags=False
    )
    protein_frag_rdmols = [
        mol
        for mol in protein_frag_rdmols
        if cdist(get_coordinates(mol), ligand_coords).min() < max_distance_to_ligand
    ]
    pocket_rdmol = reduce(Chem.CombineMols, protein_frag_rdmols)
    pocket_rdmol = Chem.AddHs(
        pocket_rdmol, explicitOnly=False, addCoords=True, addResidueInfo=False
    )

    orig_to_pocket_atom_mapping = {}
    for new_idx, atom in enumerate(pocket_rdmol.GetAtoms()):
        if atom.HasProp(_ORIG_TAG):  # heavy atoms & explicit Hs
            old_idx = atom.GetIntProp(_ORIG_TAG)
            orig_to_pocket_atom_mapping[old_idx] = new_idx

    # Round trip to reorder the added hydrgoens
    pocket_rdmol = Chem.MolFromPDBBlock(
        Chem.MolToPDBBlock(pocket_rdmol, flavor=4), sanitize=False, removeHs=False
    )
    return pocket_rdmol, orig_to_pocket_atom_mapping
