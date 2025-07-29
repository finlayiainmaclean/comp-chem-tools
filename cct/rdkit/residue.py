from rdkit import Chem


def set_residue_info(
    mol: Chem.Mol, chain_id: str = "Z", resname: str = "LIG", resnum: int = 999
):
    """Mutate mol in-place so atoms belong to same PDB residue.

    Mutates `mol` in-place so that all atoms belong to the same PDB
    residue (number `resnum`) on chain `chain_id`.

    RDKit rules / caveats
    ---------------------
    • Each Atom carries an AtomPDBResidueInfo object; create one if missing.
    • `chain_id` must be a single character (PDB allows only 1).
    • `resnum` must be 1–9999 to stay within PDB formatting limits.
    """
    # Roundtrip to PDB to set the PDB residue info.
    mol_copy = Chem.MolFromPDBBlock(
        Chem.MolToPDBBlock(mol), sanitize=False, removeHs=False
    )

    if len(chain_id) != 1:
        raise ValueError("PDB chain IDs are exactly one character")

    for at, at_copy in zip(mol.GetAtoms(), mol_copy.GetAtoms(), strict=False):
        info = at_copy.GetPDBResidueInfo()
        info.SetChainId(str(chain_id))
        info.SetResidueNumber(int(resnum))
        info.SetResidueName(str(resname))
        at.SetMonomerInfo(info)
