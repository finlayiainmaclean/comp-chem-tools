import numpy as np
from datamol import same_mol
from rdkit import Chem
from rdkit.Chem import AllChem


def sort_conformers_by_tag(mol: Chem.Mol, tag: str, ascending: bool = True) -> Chem.Mol:
    """Sort conformers in an RDKit molecule by a scalar SD tag stored on each conformer.

    Args:
    ----
        mol: RDKit Mol with multiple conformers.
        tag: The name of the SD tag to sort by.
        ascending: Sort in ascending order if True, descending if False.

    Returns:
    -------
        A new RDKit Mol with conformers sorted by the tag value.

    """
    confs = mol.GetConformers()
    values = []

    for conf in confs:
        if conf.HasProp(tag):
            val = conf.GetProp(tag)
            try:
                val = float(val)
            except ValueError:
                raise ValueError(
                    f"SD tag '{tag}' on conformer {conf.GetId()} is not a float: {val}"
                ) from None
            values.append((conf, val))
        else:
            raise KeyError(f"Conformer {conf.GetId()} is missing tag '{tag}'")

    # Sort by tag value
    values.sort(key=lambda pair: pair[1], reverse=not ascending)

    # Create a new molecule with sorted conformers
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()

    for i, (conf, _val) in enumerate(values):
        new_conf = Chem.Conformer(conf)
        new_mol.AddConformer(new_conf, assignId=True)
        # Copy all properties from original conformer, except the `tag` property
        for prop_name in conf.GetPropNames():
            prop_val = conf.GetProp(prop_name)
            new_mol.GetConformer(i).SetProp(prop_name, prop_val)

    return new_mol


def get_conformer_values(mol: Chem.Mol, tag: str) -> np.ndarray[float]:
    """Extract scalar values from SD tags on conformers.

    Args:
    ----
        mol: RDKit molecule with conformers.
        tag: Name of the SD tag to extract.

    Returns:
    -------
        Array of values from the SD tags.

    """
    confs = mol.GetConformers()
    values = []

    for conf in confs:
        if conf.HasProp(tag):
            val = conf.GetProp(tag)
            try:
                val = float(val)
            except ValueError:
                raise ValueError(
                    f"SD tag '{tag}' on conformer {conf.GetId()} is not a float: {val}"
                ) from None
            values.append((conf, val))
        else:
            raise KeyError(f"Conformer {conf.GetId()} is missing tag '{tag}'")

    values = np.array(values)
    return values


def reindex_conformers(mol: Chem.Mol) -> None:
    """Reassign conformer IDs to be sequential starting from 0."""
    # Deepcopy all conformers *before* removing them
    for i, conf in enumerate(mol.GetConformers()):
        conf.SetId(i)


def merge_conformers(
    mols: list[Chem.Mol], skip_same_mol_check: bool = False
) -> Chem.Mol:
    """Merge conformers from multiple RDKit molecules into a single
    multi-conformer molecule.

    All input molecules must have identical molecular graphs.

    Args:
    ----
        mols: List of Chem.Mol objects, each with at least one conformer.
        skip_same_mol_check: Skip validation of identical molecular graphs.

    Returns:
    -------
        A Chem.Mol with all conformers merged.

    Raises:
    ------
        ValueError if molecular graphs are not identical.

    """
    if not mols:
        raise ValueError("No molecules provided.")

    ref = Chem.Mol(mols[0])  # Reference molecule
    ref.RemoveAllConformers()

    for i, mol in enumerate(mols):
        # Sanity check: same graph
        if not skip_same_mol_check and not same_mol(mol, ref):
            raise ValueError(f"Molecule {i} differs from ref")

        for conf in mol.GetConformers():
            new_conf = Chem.Conformer(conf)
            ref.AddConformer(new_conf, assignId=True)

            # Copy conformer-level properties
            for prop in conf.GetPropNames():
                ref.GetConformer(ref.GetNumConformers() - 1).SetProp(
                    prop, conf.GetProp(prop)
                )

    return ref


def split_conformers(mol: Chem.Mol) -> list[Chem.Mol]:
    """Split a multi-conformer RDKit molecule into a list of single-conformer molecules.

    Args:
    ----
        mol: RDKit Mol with multiple conformers.

    Returns:
    -------
        List of Chem.Mol objects, each with one conformer.

    """
    single_conformers = []

    for conf in mol.GetConformers():
        new_mol = Chem.Mol(mol)  # Copy molecular graph
        new_mol.RemoveAllConformers()
        new_conf = Chem.Conformer(conf)
        new_mol.AddConformer(new_conf, assignId=True)

        # Copy conformer-level properties
        for prop in conf.GetPropNames():
            new_mol.GetConformer(0).SetProp(prop, conf.GetProp(prop))

        single_conformers.append(new_mol)

    return single_conformers


def sort_conformers_by_values(
    mol: Chem.Mol, values: list[float], ascending: bool = True
) -> Chem.Mol:
    """Sort conformers in an RDKit molecule by a list of scalar values
    (e.g., energies, RMSDs).

    Args:
    ----
        mol: RDKit Mol with multiple conformers.
        values: List of floats, one per conformer.
        ascending: Whether to sort values in ascending (True) or descending
        (False) order.

    Returns:
    -------
        A new molecule with conformers sorted by the specified values.

    """
    num_confs = mol.GetNumConformers()

    assert num_confs == len(values), f"Expected {num_confs} values, got {len(values)}."

    # Pair conformers with values and sort
    conf_value_pairs = list(zip(mol.GetConformers(), values, strict=False))
    conf_value_pairs.sort(key=lambda pair: pair[1], reverse=not ascending)

    # Create a new molecule and add sorted conformers
    new_mol = Chem.Mol(mol)  # Copy molecular graph
    new_mol.RemoveAllConformers()

    sorted_values = []
    for _i, (conf, val) in enumerate(conf_value_pairs):
        new_conf = Chem.Conformer(conf)
        new_mol.AddConformer(new_conf, assignId=True)
        sorted_values.append(val)

    return new_mol, sorted_values


def keep_conformers(mol: Chem.Mol, conf_ids: list[int], renumber_conf_ids: bool = True):
    """Remove all conformers from an RDKit molecule except those with specified IDs.

    Args:
    ----
        mol: RDKit molecule.
        conf_ids: List of conformer IDs to keep.
        renumber_conf_ids: Whether to renumber conformer IDs sequentially.

    """
    # Convert to a set for fast lookup
    keep_set = set(conf_ids)

    # Get all conformer IDs
    all_conf_ids = [conf.GetId() for conf in mol.GetConformers()]

    # Determine which to remove
    conf_ids_to_remove = [cid for cid in all_conf_ids if cid not in keep_set]

    # Remove in reverse to avoid index shifting
    for conf_id in sorted(conf_ids_to_remove, reverse=True):
        mol.RemoveConformer(conf_id)

    if renumber_conf_ids:
        reindex_conformers(mol)


def remove_conformers(
    mol: Chem.Mol, conf_ids: list[int], renumber_conf_ids: bool = True
):
    """Remove specific conformers from an RDKit molecule in-place.

    Args:
    ----
        mol: RDKit molecule.
        conf_ids: List of conformer IDs to remove.
        renumber_conf_ids: Whether to renumber conformer IDs sequentially.

    """
    # Sort in reverse to avoid index shifting
    for conf_id in sorted(conf_ids, reverse=True):
        mol.RemoveConformer(conf_id)

    if renumber_conf_ids:
        reindex_conformers(mol)


def remove_all_conformers(mol: Chem.Mol):
    """Remove all conformers from an RDKit molecule in-place.

    Args:
    ----
        mol: RDKit molecule with 0 or more conformers.

    """
    # Remove conformers in reverse to avoid index shifting
    for conf_id in reversed(range(mol.GetNumConformers())):
        mol.RemoveConformer(conf_id)


def conformer_as_mol(mol: Chem.Mol, /, cid: int) -> Chem.Mol:
    """Extract a single conformer from a multi-conformer molecule."""
    single_conf = Chem.Mol(mol)
    single_conf.RemoveAllConformers()
    single_conf.AddConformer(mol.GetConformer(cid), assignId=True)
    single_mol = Chem.Mol(single_conf)
    return single_mol


def embed(
    mol: Chem.Mol, num_confs: int = 100, rmsd_threshold: float | None = None
) -> Chem.Mol:
    """Embed `nconf` ETKDGv3 conformers and MMFF94‑optimise them in place."""
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if rmsd_threshold:
        params.pruneRmsThresh = rmsd_threshold
    params.numThreads = 0  # use all cores for embedding
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)

    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")
    for cid in range(mol.GetNumConformers()):
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
        ff.Minimize(maxIts=10000)
    return mol
