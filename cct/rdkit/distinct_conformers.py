import numpy as np
from rdkit.Chem import rdMolAlign

from cct.ase_calcs import CalculatorFactory
from cct.rdkit.conformers import keep_conformers
from cct.rdkit.io import rdkit_from_xyz


def calculate_rmsd(mol, conf_id1, conf_id2):
    """Calculate RMSD between two conformers using RDKit"""
    # Use RDKit's best alignment function (includes optimization)
    conf_id1 = int(conf_id1)
    conf_id2 = int(conf_id2)
    rmsd = rdMolAlign.GetBestRMS(mol, mol, conf_id1, conf_id2)
    return rmsd


def calculate_dE(energy1, energy2):
    """Calculate energy difference"""
    return abs(energy1 - energy2)


def calculate_deltaBe(mol, conf_id1, conf_id2):
    """Calculate difference in rotational constants (percentage like CREST's cregen)"""
    # Get coordinates for both conformers
    conf_id1 = int(conf_id1)
    conf_id2 = int(conf_id2)
    conf1 = mol.GetConformer(conf_id1)
    conf2 = mol.GetConformer(conf_id2)

    # Calculate moments of inertia for both conformers
    def get_moments_of_inertia(conf):
        # Get atomic masses and coordinates
        masses = np.array([mol.GetAtomWithIdx(i).GetMass() for i in range(mol.GetNumAtoms())])
        coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])

        # Calculate center of mass
        total_mass = np.sum(masses)
        com = np.sum(masses[:, np.newaxis] * coords, axis=0) / total_mass

        # Translate coordinates to center of mass
        coords_com = coords - com

        # Calculate moment of inertia tensor
        I_tensor = np.zeros((3, 3))
        for i, (mass, pos) in enumerate(zip(masses, coords_com)):
            x, y, z = pos
            I_tensor[0, 0] += mass * (y**2 + z**2)  # Ixx
            I_tensor[1, 1] += mass * (x**2 + z**2)  # Iyy
            I_tensor[2, 2] += mass * (x**2 + y**2)  # Izz
            I_tensor[0, 1] -= mass * x * y  # Ixy
            I_tensor[0, 2] -= mass * x * z  # Ixz
            I_tensor[1, 2] -= mass * y * z  # Iyz

        # Make tensor symmetric
        I_tensor[1, 0] = I_tensor[0, 1]
        I_tensor[2, 0] = I_tensor[0, 2]
        I_tensor[2, 1] = I_tensor[1, 2]

        # Get eigenvalues (principal moments of inertia)
        eigenvals = np.linalg.eigvals(I_tensor)
        eigenvals = np.sort(eigenvals)  # Sort in ascending order

        return eigenvals

    # Calculate moments for both conformers
    moments1 = get_moments_of_inertia(conf1)
    moments2 = get_moments_of_inertia(conf2)

    # Convert moments of inertia to rotational constants (in MHz, like CREST)
    # B = h / (8 * pi^2 * I) where I is in kg⋅m²
    # RDKit coordinates are in Angstrom, masses in amu
    h = 6.62607015e-34  # J⋅s
    amu_to_kg = 1.66053906660e-27  # kg
    angstrom_to_m = 1e-10  # m

    # Conversion factor to MHz
    conversion = h / (8 * np.pi**2) * 1e-6 / (amu_to_kg * angstrom_to_m**2)

    # Handle zero moments of inertia (linear molecules)
    def moments_to_rotconst(moments):
        rotconst = np.zeros_like(moments)
        for i, moment in enumerate(moments):
            if moment > 1e-10:  # avoid division by zero
                rotconst[i] = conversion / moment
            else:
                rotconst[i] = 0.0
        return rotconst

    rotconst1 = moments_to_rotconst(moments1)
    rotconst2 = moments_to_rotconst(moments2)

    # Calculate percentage differences for each rotational constant (like CREST)
    relative_diffs = []
    for i in range(len(rotconst1)):
        if rotconst1[i] > 1e-6 and rotconst2[i] > 1e-6:  # only for non-zero constants
            # Calculate relative difference as percentage
            avg_rotconst = (rotconst1[i] + rotconst2[i]) / 2
            rel_diff = abs(rotconst1[i] - rotconst2[i]) / avg_rotconst
            relative_diffs.append(rel_diff)

    if relative_diffs:
        # Use average relative difference as percentage (0-100 range)
        delta_Be = np.mean(relative_diffs) * 100
    else:
        delta_Be = 0.0

    return delta_Be


def is_distinct_conformer(
    mol,
    conf_id1,
    conf_id2,
    energy1,
    energy2,
    energy_threshold=0.002174,  # 0.05 kcal/mol in eV
    rotation_threshold=1,  # % change of rotation constants,
    rmsd_threshold=0.125,  # Ang
):
    dE = calculate_dE(energy1, energy2)
    rot_diff = calculate_deltaBe(mol, conf_id1, conf_id2)
    rmsd = calculate_rmsd(mol, conf_id1, conf_id2)

    if dE > energy_threshold or rot_diff > rotation_threshold or rmsd > rmsd_threshold:
        return True
    return False


def get_distinct_conformers(
    mol,
    energies,
    energy_threshold=0.002174,  # 0.05 kcal/mol in eV
    rotation_threshold=1,  # % change of rotation constants,
    rmsd_threshold=0.125,  # Ang
):
    old_len = len(energies)
    sorted_idxs = np.argsort(energies)
    conformers_idxs = []
    conformers_idxs.append(sorted_idxs[0])

    for query_idx in sorted_idxs[1:]:
        mask = [
            is_distinct_conformer(
                mol,
                conf_id1=ref_idx,
                conf_id2=query_idx,
                energy1=energies[ref_idx],
                energy2=energies[query_idx],
                energy_threshold=energy_threshold,
                rotation_threshold=rotation_threshold,
                rmsd_threshold=rmsd_threshold,
            )
            for ref_idx in conformers_idxs
        ]
        if sum(mask) == len(mask):
            conformers_idxs.append(query_idx)
    conformers_idxs = np.array(conformers_idxs)

    print(f"Filtered {old_len} structures to {len(conformers_idxs)} distinct conformers")

    mol = keep_conformers(mol, conf_ids=conformers_idxs)
    energies = energies[conformers_idxs]

    return mol, energies


if __name__ == "__main__":
    mol = rdkit_from_xyz("data/ensemble.xyz", charge=0)
    calcs = CalculatorFactory()
    energies = calcs.singlepoint_mol(mol, method="GFN2-xTB")
    get_distinct_conformers(mol, energies)
