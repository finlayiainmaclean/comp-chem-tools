from rdkit import Chem


class ImplicitHydrogenError(ValueError):
    pass


def assert_all_hydrogens_explicit(mol: Chem.Mol):
    """Ensure that *every* hydrogen in `mol` is represented as an explicit atom.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Chem.Mol that is expected to have *no* implicit hydrogens.

    Raises
    ------
    ValueError
        If any atom retains implicit hydrogens.

    """
    # Make sure the atom property caches are up-to-date
    mol.UpdatePropertyCache(strict=False)

    for atom in mol.GetAtoms():
        n_implicit = atom.GetNumImplicitHs()
        if n_implicit:
            raise ImplicitHydrogenError(
                f"Atom {atom.GetIdx()} ({atom.GetSymbol()}) still has "
                f"{n_implicit} implicit hydrogen{'s' if n_implicit > 1 else ''}."
            )
