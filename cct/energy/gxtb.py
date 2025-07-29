import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem

from cct.energy.consts import EH2KCALMOL


def parse_energy(filename: str) -> float:
    """Parse energy from gxtb output file."""
    energy = None
    inside_energy_block = False

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "$energy":
                inside_energy_block = True
                continue
            if line == "$end" and inside_energy_block:
                break
            if inside_energy_block:
                # Parse line like:
                # 1   -76.43725029375892   -6.03098024186477  99.9 99.9 99.9
                parts = line.split()
                if len(parts) >= 2:
                    energy = float(parts[1])
                    break
    return energy * EH2KCALMOL


def gxtb_singlepoint(mol: Chem.Mol, use_docker: bool = True) -> float:
    """Calculate single-point energy using gxtb."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_input_file = tmpdir / "input.xyz"
        tmp_energy_file = tmpdir / "energy"
        Chem.MolToXYZFile(mol, tmp_input_file)

        cmd = "gxtb -c input.xyz"
        if use_docker:
            cmd = f"docker run --rm -v {tmpdir!s}:/data -w /data gxtb {cmd}"

        subprocess.run(cmd, cwd=tmpdir, check=True, shell=True)

        energy = parse_energy(tmp_energy_file)

    return energy


def gxtb_optimise(mol: Chem.Mol, use_docker: bool = True) -> float:
    """Optimize molecular geometry using gxtb."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_input_file = tmpdir / "input.xyz"
        tmp_energy_file = tmpdir / "energy"
        tmp_output_file = tmpdir / "xtbopt.xyz"
        Chem.MolToXYZFile(mol, tmp_input_file)

        cmd = 'xtb input.xyz --driver "gxtb -grad -c xtbdriver.xyz" --opt'
        if use_docker:
            cmd = f"docker run --rm -v {tmpdir!s}:/data -w /data gxtb {cmd}"

        subprocess.run(cmd, cwd=tmpdir, check=True, shell=True)
        energy = parse_energy(tmp_energy_file)
        mol_opt = Chem.MolFromXYZFile(str(tmp_output_file))

    return mol_opt, energy
