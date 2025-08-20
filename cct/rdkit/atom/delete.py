from rdkit import Chem


def delete_atom(mol, atom_idx):
    """Delete a specific bond object"""
    editable_mol = Chem.EditableMol(mol)
    editable_mol.RemoveAtom(atom_idx)
    return editable_mol.GetMol()
