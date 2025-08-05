import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from rdkit import Chem

from cct.rdkit.io import from_xyz, to_xyz


class CRESTError(Exception):
    pass


def run_metadynamics(
    mol: Chem.Mol,
    charge: int = 0,
    multiplicity: int = 1,
    num_cores: int = 1,
    method: Literal["gfn2", "gfn1", "gfn0"] = "gfn2",
    quick: bool = False,
):
    """Run `crest singlepoint on an ensemble."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        tmp_input_xyz = tmpdir / "mol.xyz"
        tmp_mid_xyz = tmpdir / "crest_ensemble.xyz"
        tmp_output_xyz = tmpdir / "crest_conformers.xyz"

        to_xyz(mol, tmp_input_xyz)

        cmd = [
            "crest",
            "--mdopt",
            str(tmp_input_xyz),
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
            "--method",
            method,
            "-T",
            str(num_cores),
        ]
        subprocess.run(cmd, cwd=tmpdir)

        if not tmp_mid_xyz.exists():
            raise CRESTError("CREST optimisation failed")

        cmd = [
            "crest",
            str(tmp_mid_xyz),
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
            "--method",
            method,
            "-T",
            str(num_cores),
        ]

        if quick:
            cmd.append("--quick")

        subprocess.run(cmd, cwd=tmpdir)

        if not tmp_output_xyz.exists():
            raise CRESTError("CREST metadynamics failed")

        mol = from_xyz(tmp_output_xyz, ref_mol=mol)

    return mol


def run_screen(
    mol: Chem.Mol,
    screen_type: Literal["singlepoint", "optimise"],
    charge: int = 0,
    multiplicity: int = 1,
    num_cores: int = 1,
    method: Literal["gfn2", "gfn1", "gfn0"] = "gfn2",
    rmsd_threshold: float = 0.125,
    conformer_energy_window: float = 0.05,
    energy_window: float = 6.0,
    rotational_threshold: float = 0.01,
) -> tuple[Chem.Mol, np.ndarray[float]]:
    """Run CREST screening on molecular conformers.

    Args:
    ----
        mol: RDKit molecule with conformers.
        screen_type: Type of screening to perform.
        charge: Molecular charge.
        multiplicity: Spin multiplicity.
        num_cores: Number of CPU cores to use.
        method: QM method for calculations.
        rmsd_threshold: RMSD threshold for conformer filtering.
        conformer_energy_window: Energy window for conformer filtering.
        energy_window: Overall energy window.
        rotational_threshold: Rotational threshold for filtering.

    Returns:
    -------
        Tuple of processed molecule and energy array.

    """
    match screen_type:
        case "singlepoint":
            mol = run_singlepoint(
                mol,
                multiplicity=multiplicity,
                charge=charge,
                num_cores=num_cores,
                method=method,
            )
        case "optimise":
            mol = run_optimisation(
                mol,
                multiplicity=multiplicity,
                charge=charge,
                num_cores=num_cores,
                method=method,
            )

    mol = run_cregen(
        mol,
        multiplicity=multiplicity,
        charge=charge,
        num_cores=num_cores,
        rmsd_threshold=rmsd_threshold,
        conformer_energy_window=conformer_energy_window,
        energy_window=energy_window,
        rotational_threshold=rotational_threshold,
    )

    energies = np.array([conf.GetDoubleProp("Energy") for conf in mol.GetConformers()])

    return mol, energies


def run_singlepoint(
    mol: Chem.Mol,
    charge: int = 0,
    multiplicity: int = 1,
    num_cores: int = 1,
    method: Literal["gfn2", "gfn1", "gfn0"] = "gfn2",
):
    """Run `crest optimisation on an ensemble."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        tmp_input_xyz = tmpdir / "input.xyz"
        tmp_output_xyz = tmpdir / "crest_ensemble.xyz"

        to_xyz(mol, tmp_input_xyz)

        cmd = [
            "crest",
            "-for",
            str(tmp_input_xyz),
            "--prop",
            "singlepoint",
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
            "---method",
            method,
            "-T",
            str(num_cores),
        ]
        subprocess.run(cmd, cwd=tmpdir)

        if not tmp_output_xyz.exists():
            raise CRESTError("CREST singlepoint failed")

        mol = from_xyz(tmp_output_xyz, ref_mol=mol)

    return mol


def run_optimisation(
    mol: Chem.Mol,
    charge: int = 0,
    multiplicity: int = 1,
    num_cores: int = 1,
    method: Literal["gfn2", "gfn1", "gfn0"] = "gfn2",
):
    """Run `crest singlepoint on an ensemble."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        tmp_input_xyz = tmpdir / "input.xyz"
        tmp_output_xyz = tmpdir / "crest_ensemble.xyz"

        to_xyz(mol, tmp_input_xyz)

        cmd = [
            "crest",
            "--mdopt",
            str(tmp_input_xyz),
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
            "--method",
            method,
            "-T",
            str(num_cores),
        ]
        subprocess.run(cmd, cwd=tmpdir)

        if not tmp_output_xyz.exists():
            raise CRESTError("CREST optimisation failed")

        mol = from_xyz(tmp_output_xyz, ref_mol=mol)
    return mol


def run_cregen(
    mol: Chem.Mol,
    charge: int = 0,
    multiplicity: int = 1,
    rmsd_threshold: float = 0.125,
    conformer_energy_window: float = 0.05,
    energy_window: float = 6.0,
    rotational_threshold: float = 0.01,
    num_cores: int = 1,
):
    """Run `crest --screen` and return the resulting `screen.xyz` path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        tmp_input_xyz = tmpdir / "input.xyz"
        tmp_output_xyz = tmpdir / "crest_ensemble.xyz"

        to_xyz(mol, tmp_input_xyz)

        cmd = [
            "crest",
            str(tmp_input_xyz),
            "--cregen",
            str(tmp_input_xyz),
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
            "--rthr",
            str(rmsd_threshold),
            "--ethr",
            str(conformer_energy_window),
            "--ewin",
            str(energy_window),
            "--bthr",
            str(rotational_threshold),
            "-T",
            str(num_cores),
            "--notopo",
        ]
        subprocess.run(cmd, cwd=tmpdir)

        if not tmp_output_xyz.exists():
            raise CRESTError("CREST cregen failed")

        mol = from_xyz(tmp_output_xyz, ref_mol=mol)

    return mol
