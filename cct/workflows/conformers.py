from multiprocessing import cpu_count
from typing import Literal, get_args

import click
import numpy as np
from rdkit import Chem

from cct.ase_calcs import QM_METHODS, CalculatorFactory
from cct.rdkit.conformers import embed, keep_conformers, sort_conformers_by_values
from cct.rdkit.crest import run_metadynamics, run_screen

RDKIT_MODES = Literal["RECKLESS", "RAPID"]
CREST_MODES = Literal["METICULOUS", "CAREFUL"]
MODES = Literal["METICULOUS", "CAREFUL", "RECKLESS", "RAPID"]


def _generate_conformers_crest(
    mol,
    mode: CREST_MODES,
    multiplicity: int = 1,
    charge: int = 0,
    num_cores: int = None,
):
    """Generate conformers using CREST method."""
    if num_cores is None:
        num_cores = cpu_count()

    mol = run_metadynamics(
        mol,
        charge=charge,
        multiplicity=multiplicity,
        quick=mode == "CAREFUL",
        num_cores=num_cores,
    )
    return mol


def _generate_conformers_rdkit(
    mol,
    mode: RDKIT_MODES,
    charge: int | None = None,
    multiplicity: int = 1,
    num_cores: int = None,
):
    """Generate conformers using RDKit method."""
    if not charge:
        charge = Chem.GetFormalCharge(mol)

    if num_cores is None:
        num_cores = cpu_count()

    # 1. Embed using RDKit
    match mode:
        case "RECKLESS":
            rmsd_threshold = 0.25
            mol = embed(mol, num_confs=100, rmsd_threshold=rmsd_threshold)
            initial_energy_window = 10.0
            final_energy_window = 5.0
            max_conformers_sp_sqm = 50
            max_conformers_opt_sqm = 20

        case "RAPID":
            rmsd_threshold = 0.1
            mol = embed(mol, num_confs=300, rmsd_threshold=rmsd_threshold)
            initial_energy_window = 20.0
            final_energy_window = 10.0
            max_conformers_sp_sqm = 100
            max_conformers_opt_sqm = 50

    click.echo(f"{mol.GetNumConformers()} conformers")

    click.echo("Running SQM singlepoint")
    mol, energies = run_screen(
        mol,
        screen_type="singlepoint",
        multiplicity=multiplicity,
        charge=charge,
        num_cores=num_cores,
        energy_window=initial_energy_window,
        rmsd_threshold=rmsd_threshold,
    )
    conf_ids_to_keep = np.argsort(energies)[:max_conformers_sp_sqm]
    click.echo(f"{mol.GetNumConformers()} conformers")
    keep_conformers(mol, conf_ids_to_keep)
    click.echo(f"{mol.GetNumConformers()} conformers")

    click.echo("Running SQM optimisation")
    mol, energies = run_screen(
        mol,
        screen_type="optimise",
        multiplicity=multiplicity,
        charge=charge,
        num_cores=num_cores,
        energy_window=final_energy_window,
        rmsd_threshold=rmsd_threshold,
    )
    conf_ids_to_keep = np.argsort(energies)[:max_conformers_opt_sqm]
    click.echo(f"{mol.GetNumConformers()} conformers")
    keep_conformers(mol, conf_ids_to_keep)
    click.echo(f"{mol.GetNumConformers()} conformers")

    return mol


def generate_conformers(
    mol: Chem.Mol,
    /,
    *,
    mode: MODES = "rapid",
    qm_method: QM_METHODS = "MACE_MEDIUM",
    charge: int = 0,
    multiplicity: int = 1,
    num_cores: int | None = None,
):
    """Generate molecular conformers using various computational methods.

    Args:
    ----
        mol: RDKit molecule object.
        mode: Conformer generation mode (reckless, rapid, careful, meticulous).
        qm_method: Quantum mechanical method for calculations.
        charge: Molecular charge.
        multiplicity: Spin multiplicity.
        num_cores: Number of CPU cores to use (None for all available).

    Returns:
    -------
        Tuple of optimized molecule and solvated energies.

    """
    if num_cores is None:
        num_cores = cpu_count()

    # Set charge if not provided
    if charge is None:
        charge = Chem.GetFormalCharge(mol)

    # Generate conformers based on mode
    mode = mode.upper()
    if mode in get_args(CREST_MODES):
        mol = _generate_conformers_crest(
            mol,
            mode=mode,
            multiplicity=multiplicity,
            charge=charge,
            num_cores=num_cores,
        )
    elif mode in get_args(RDKIT_MODES):
        mol = _generate_conformers_rdkit(
            mol,
            mode=mode,
            multiplicity=multiplicity,
            charge=charge,
            num_cores=num_cores,
        )
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
    click.echo(f"Running QM singlepoint with {mol.GetNumConformers()} conformers")

    energies = calcs.singlepoint(
        mol, method=qm_method, multiplicity=multiplicity, charge=charge
    )
    conf_ids_to_keep = np.argsort(energies)[:max_conformers_qm]
    keep_conformers(mol, conf_ids_to_keep)

    click.echo(f"Running QM optimisation with {mol.GetNumConformers()} conformers")

    # Optimization
    mol, gas_phase_energies = calcs.optimise(
        mol, method=qm_method, multiplicity=multiplicity, charge=charge
    )

    # Solvation energy
    solvation_energies = calcs.solvation_energy(
        mol, multiplicity=multiplicity, charge=charge
    )

    # Combined energies
    solvated_energies = gas_phase_energies + solvation_energies
    mol, solvated_energies = sort_conformers_by_values(mol, solvated_energies)
    return mol, solvated_energies


@click.command()
@click.option(
    "--mode",
    type=click.Choice(get_args(MODES), case_sensitive=False),
    default="reckless",
    help="Conformer generation mode",
)
@click.option("--input", required=True, help="Input molecule file")
@click.option("--output", type=click.Path(), help="Output file path for results")
@click.option(
    "--multiplicity", type=int, default=1, help="Spin multiplicity of the molecule"
)
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

    click.echo(f"Using {num_cores} CPU cores")
    click.echo(f"Mode: {mode}")
    click.echo(f"Input: {input}")
    click.echo(f"Multiplicity: {multiplicity}")
    click.echo(f"QM Method: {qm_method}")

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
    click.echo(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
