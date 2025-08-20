from rdkit import Chem


def delete_bond(mol, bond):
    """Delete a specific bond object"""
    editable_mol = Chem.EditableMol(mol)
    begin_idx = bond.GetBeginAtomIdx()
    end_idx = bond.GetEndAtomIdx()
    editable_mol.RemoveBond(begin_idx, end_idx)
    return editable_mol.GetMol()
