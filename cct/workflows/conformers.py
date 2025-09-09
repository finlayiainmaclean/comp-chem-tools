from multiprocessing import cpu_count
from typing import Literal, get_args

import click
import numpy as np
from rdkit import Chem

from cct.ase_calcs import QM_METHODS, CalculatorFactory
from cct.crest import run_metadynamics
from cct.rdkit.conformers import embed, keep_conformers, remove_high_energy_conformers, sort_conformers_by_values
from cct.rdkit.distinct_conformers import get_distinct_conformers

RDKIT_MODES = Literal["RECKLESS", "RAPID"]
CREST_MODES = Literal["METICULOUS", "CAREFUL"]
MODES = Literal["METICULOUS", "CAREFUL", "RECKLESS", "RAPID"]

USE_CREST = False


def _generate_conformers_crest(
    mol,
    mode: CREST_MODES,
    multiplicity: int = 1,
    charge: int = 0,
    solvent: Literal["water"] | None = None,
    num_cores: int | None = None,
):
    """Generate conformers using CREST method."""
    if num_cores is None:
        num_cores = cpu_count()

    mol = run_metadynamics(mol, charge=charge, multiplicity=multiplicity, quick=mode == "CAREFUL", solvent=solvent)
    return mol


def _generate_conformers_rdkit(
    mol, mode: RDKIT_MODES, charge: int | None = None, multiplicity: int = 1, solvent: Literal["water"] | None = None
):
    """Generate conformers using RDKit method."""
    if not charge:
        charge = Chem.GetFormalCharge(mol)

    # 1. Embed using RDKit
    mode = mode.upper()
    match mode:
        case "RECKLESS":
            rmsd_threshold = 0.25
            initial_energy_window = 10.0
            final_energy_window = 5.0
            max_conformers_sp_sqm = 50
            max_conformers_opt_sqm = 20
            mol = embed(mol, num_confs=100, rmsd_threshold=rmsd_threshold)

        case "RAPID":
            rmsd_threshold = 0.1
            initial_energy_window = 20.0
            final_energy_window = 10.0
            max_conformers_sp_sqm = 100
            max_conformers_opt_sqm = 50
            mol = embed(mol, num_confs=300, rmsd_threshold=rmsd_threshold)

        case _:
            raise NotImplementedError(f"{mode} not a valid mode.")

    print("Running SQM singlepoint")

    calcs = CalculatorFactory()
    energies = calcs.singlepoint_mol(mol, method="GFN2-xTB", charge=charge, multiplicity=multiplicity, solvent=solvent)
    mol, energies = remove_high_energy_conformers(mol, energies=energies, energy_window=initial_energy_window)
    mol, energies = get_distinct_conformers(mol, energies=energies, rmsd_threshold=rmsd_threshold)

    conf_ids_to_keep = np.argsort(energies)[:max_conformers_sp_sqm]
    keep_conformers(mol, conf_ids=conf_ids_to_keep)

    print("Running SQM optimisation")
    mol, energies = calcs.optimise_mol(
        mol, method="GFN2-xTB", charge=charge, multiplicity=multiplicity, solvent=solvent
    )
    mol, energies = remove_high_energy_conformers(mol, energies=energies, energy_window=final_energy_window)
    mol, energies = get_distinct_conformers(mol, energies=energies, rmsd_threshold=rmsd_threshold)

    conf_ids_to_keep = np.argsort(energies)[:max_conformers_opt_sqm]
    keep_conformers(mol, conf_ids=conf_ids_to_keep)
    print(f"{mol.GetNumConformers()} conformers")

    return mol


def generate_conformers(
    mol: Chem.Mol,
    /,
    *,
    mode: MODES = "rapid",
    qm_method: QM_METHODS = "AIMNet2",
    charge: int | None = None,
    multiplicity: int = 1,
    solvent: Literal["water"] | None = None,
):
    """Generate molecular conformers using various computational methods.

    Args:
    ----
        mol: RDKit molecule object.
        mode: Conformer generation mode (reckless, rapid, careful, meticulous).
        qm_method: Quantum mechanical method for calculations.
        charge: Molecular charge.
        multiplicity: Spin multiplicity.

    Returns:
    -------
        Tuple of optimized molecule and solvated energies.

    """

    # Set charge if not provided
    if charge is None:
        charge = Chem.GetFormalCharge(mol)

    # Generate conformers based on mode
    mode = mode.upper()
    if mode in get_args(CREST_MODES):
        mol = _generate_conformers_crest(mol, mode=mode, multiplicity=multiplicity, charge=charge, solvent=solvent)
    elif mode in get_args(RDKIT_MODES):
        mol = _generate_conformers_rdkit(mol, mode=mode, multiplicity=multiplicity, charge=charge, solvent=solvent)
    else:
        raise ValueError(f"Unknown mode: {mode}", err=True)

    # Determine max conformers for QM based on mode
    max_conformers_qm_map = {
        "RECKLESS": 20,
        "RAPID": 50,
        "CAREFUL": 50,
        "METICULOUS": 150,
    }
    max_conformers_qm = max_conformers_qm_map[mode]

    # QM calculations
    calcs = CalculatorFactory()

    # Single point calculations
    print(f"Running QM singlepoint with {mol.GetNumConformers()} conformers")

    energies = calcs.singlepoint_mol(
        mol,
        method=qm_method,
        multiplicity=multiplicity,
        charge=charge,
    )
    conf_ids_to_keep = np.argsort(energies)[:max_conformers_qm]
    keep_conformers(mol, conf_ids=conf_ids_to_keep)

    print(f"Running QM optimisation with {mol.GetNumConformers()} conformers")

    # Optimization
    mol, energies = calcs.optimise_mol(
        mol,
        method=qm_method,
        multiplicity=multiplicity,
        charge=charge,
    )
    if solvent == "water":
        # Solvation energy
        solvation_energies = calcs.solvation_energy_mol(
            mol, multiplicity=multiplicity, charge=charge, method="GFN2-xTB"
        )

        # Combined energies
        energies = energies + solvation_energies
    elif solvent is not None:
        raise ValueError("Only water solvent supported")

    mol, energies = sort_conformers_by_values(mol, values=energies)
    return mol, energies


@click.command()
@click.option(
    "--mode",
    type=click.Choice(get_args(MODES), case_sensitive=False),
    default="reckless",
    help="Conformer generation mode",
)
@click.option("--input", required=True, help="Input molecule file")
@click.option("--output", type=click.Path(), help="Output file path for results")
@click.option("--multiplicity", type=int, default=1, help="Spin multiplicity of the molecule")
@click.option(
    "--qm_method",
    type=click.Choice(get_args(QM_METHODS), case_sensitive=False),
    default="MACE_SMALL",
    help="QM method to use for calculations",
)
@click.option(
    "--charge",
    type=int,
    default=0,
    help="Molecular charge (if not specified, will be calculated from formal charge)",
)
@click.option(
    "--num_cores",
    type=int,
    default=-1,
    help="Number of CPU cores to use (default: all available cores)",
)
def main(mode, input, output, multiplicity, qm_method, charge, num_cores):
    """Generate molecular conformers using RDKit and quantum mechanical methods.

    This tool generates conformers for a given molecule using different levels \
of accuracy:
    - RECKLESS: Fast, lower accuracy
    - RAPID: Moderate speed and accuracy
    - CAREFUL: Slower, higher accuracy
    - METICULOUS: Slowest, highest accuracy
    """
    # Set default number of cores
    if num_cores == -1:
        num_cores = cpu_count()

    print(f"Using {num_cores} CPU cores")
    print(f"Mode: {mode}")
    print(f"Input: {input}")
    print(f"Multiplicity: {multiplicity}")
    print(f"QM Method: {qm_method}")

    # Parse molecule
    mol = Chem.MolFromMolFile(input)
    if mol is None:
        raise ValueError("Invalid molecule file")
    mol = Chem.AddHs(mol, addCoords=True)

    mol, solvated_energies = generate_conformers(
        mol, mode=mode, qm_method=qm_method, charge=charge, multiplicity=multiplicity
    )

    # Save as SDF file
    writer = Chem.SDWriter(output)
    for conf_id in range(mol.GetNumConformers()):
        conf = mol.GetConformer(conf_id)
        # Copy all conformer properties to molecule properties
        for prop_name in conf.GetPropNames():
            mol.SetProp(prop_name, conf.GetProp(prop_name))

        mol.SetProp("Energy", str(solvated_energies[conf_id]))
        writer.write(mol, confId=conf_id)
    writer.close()
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
