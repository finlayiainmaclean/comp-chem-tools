import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Final, List

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
from ase.units import Bohr, Hartree


def check_gxtb():
    """Check if gxtb is installed and required files are present.

    Returns
    -------
        bool: True if gxtb is available and all required files exist.

    """
    home_dir = Path.home()
    required_home_files = [".gxtb", ".eeq", ".basisq"]
    # Check if required files exist in home directory
    home_files_exist = all(
        (home_dir / filename).exists() for filename in required_home_files
    )
    gxtb_in_path = shutil.which("gxtb") is not None
    return gxtb_in_path and home_files_exist


def download_gxtb():
    """Download and install gxtb binary and parameter files.

    Downloads the g-xtb package from GitHub, extracts it, and copies the
    binary to /usr/local/bin and parameter files to the user's home directory.
    """
    # Download and extract g-xtb
    download_url: Final = (
        "https://github.com/grimme-lab/g-xtb/archive/refs/tags/v1.0.0.tar.gz"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        tarball_path = temp_path / "gxtb.tar.gz"

        # Download the tarball
        print(f"Downloading g-xtb from {download_url}...")
        urllib.request.urlretrieve(download_url, tarball_path)

        # Extract the tarball
        print("Extracting archive...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(temp_path)

        # Find the extracted directory
        extracted_dir = temp_path / "g-xtb-1.0.0"
        gxtb_binary = extracted_dir / "binary" / "gxtb"
        local_bin_binary = Path("/usr/local/bin") / "gxtb"
        shutil.copyfile(src=gxtb_binary, dst=local_bin_binary)
        local_bin_binary.chmod(0o755)  # Make executable

        home_dir = Path.home()
        required_home_files = [".gxtb", ".eeq", ".basisq"]
        for _home_file in required_home_files:
            extracted_file = extracted_dir / "parameters" / _home_file
            home_file = home_dir / _home_file
            shutil.copyfile(src=extracted_file, dst=home_file)


def check_and_download_gxtb():
    """Check if gxtb is available and download it if not.

    Convenience function that checks for gxtb installation and downloads
    it automatically if not found.
    """
    if not check_gxtb():
        download_gxtb()


class gxTBCalculator(Calculator):
    """ASE Calculator interface for gxtb."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        charge: int = 0,
        multiplicity: int = 1,
        **kwargs,
    ):
        """Initialize gxtb calculator.

        Args:
        ----
            charge: Molecular charge
            multiplicity: Spin multiplicity (1=singlet, 2=doublet, 3=triplet, etc.)
                         (default: None, auto-determined)
            **kwargs: Additional keyword arguments passed to parent class

        """
        super().__init__(**kwargs)
        self.charge = charge
        self.multiplicity = multiplicity

        check_and_download_gxtb()

    def calculate(
        self,
        atoms: Atoms = None,
        properties: List[str] = None,
        system_changes: List[str] = all_changes,
    ):
        """Run gxtb calculation."""
        if properties is None:
            properties = ["energy"]
        if atoms is not None:
            self.atoms = atoms.copy()

        super().calculate(atoms, properties, system_changes)

        # Create temporary directory for calculation
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write structure file
            xyz_file = tmpdir / "structure.xyz"
            write(xyz_file, self.atoms)

            # Write control files if specified
            self._write_control_files(tmpdir)

            # Calculate energy
            if "energy" in properties:
                energy = self._calculate_energy(tmpdir, xyz_file)
                self.results["energy"] = energy

            # Calculate forces (from gradients)
            if "forces" in properties:
                forces = self._calculate_forces(tmpdir, xyz_file)
                self.results["forces"] = forces

    def _write_control_files(self, tmpdir: Path):
        """Write .CHRG and .UHF control files if specified."""
        if self.charge is not None:
            chrg_file = tmpdir / ".CHRG"
            with open(chrg_file, "w") as f:
                f.write(f"{self.charge}\n")

        if self.multiplicity is not None:
            # Convert multiplicity to UHF (number of unpaired electrons)
            # Multiplicity = 2S + 1, so UHF = S * 2 = multiplicity - 1
            uhf = self.multiplicity - 1
            uhf_file = tmpdir / ".UHF"
            with open(uhf_file, "w") as f:
                f.write(f"{uhf}\n")

    def _calculate_energy(self, tmpdir: Path, xyz_file: Path) -> float:
        """Calculate energy using gxtb."""
        # Build command for energy calculation
        cmd = ["gxtb", "-c", str(xyz_file)]

        # Run calculation
        try:
            subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"gxtb energy calculation failed: {e.stderr}") from e

        # Parse energy from output
        energy_file = tmpdir / "energy"
        if energy_file.exists():
            return self._parse_energy(energy_file)
        else:
            raise RuntimeError("Energy file not found after gxtb calculation")

    def _calculate_forces(self, tmpdir: Path, xyz_file: Path) -> np.ndarray:
        """Calculate forces using gxtb gradients."""
        # Build command for gradient calculation
        cmd = ["gxtb", "-grad", "-c", str(xyz_file)]

        # Run calculation
        try:
            subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"gxtb gradient calculation failed: {e.stderr}") from e

        # Parse gradients from gradient file
        grad_file = tmpdir / "gradient"
        if grad_file.exists():
            gradients = self._parse_gradients(grad_file)
            # Convert gradients to forces (forces = -gradients)
            # Also convert from Hartree/Bohr to eV/Angstrom
            forces = -gradients * (Hartree / Bohr)
            return forces
        else:
            raise RuntimeError("Gradient file not found after gxtb calculation")

    def _parse_energy(self, filename: Path) -> float:
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

        if energy is None:
            raise RuntimeError("Could not parse energy from gxtb output")

        # Convert from Hartree to eV (ASE default energy unit)
        return energy * Hartree

    def _parse_gradients(self, filename: Path) -> np.ndarray:
        """Parse gradients from gxtb gradient file."""
        with open(filename) as f:
            content = f.read()

        lines = content.strip().split("\n")
        gradients = []

        # Look for lines with scientific notation (gradients)
        for line in lines:
            line = line.strip()
            if "D+" in line or "D-" in line:
                # Replace D with E for proper float parsing
                line = line.replace("D+", "E+").replace("D-", "E-")
                parts = line.split()

                try:
                    grad_components = [float(x) for x in parts]
                    gradients.extend(grad_components)
                except ValueError:
                    continue

        if not gradients:
            raise RuntimeError("No gradients found in gradient file")

        # Reshape to (n_atoms, 3)
        gradients = np.array(gradients)
        n_atoms = len(gradients) // 3
        gradients = gradients.reshape(n_atoms, 3)

        return gradients


# Example usage and testing
if __name__ == "__main__":
    from ase import Atoms
    from ase.io import write
    from ase.optimize import BFGS

    # Create a water molecule for testing
    water = Atoms(
        "H2O", positions=[(0.75, 0.0, 0.44), (0.0, 0.0, -0.17), (-0.75, 0.0, 0.44)]
    )

    # Set up the calculator
    calc = gxTBCalculator(charge=0, multiplicity=1)  # Neutral singlet
    water.calc = calc

    # Calculate energy
    print(f"Energy: {water.get_potential_energy():.6f} eV")

    # Calculate forces
    forces = water.get_forces()
    print("Forces (eV/Å):")
    for i, force in enumerate(forces):
        print(f"  Atom {i + 1}: {force[0]:10.6f} {force[1]:10.6f} {force[2]:10.6f}")

    # Example optimization
    print("\nRunning geometry optimization...")
    opt = BFGS(water, trajectory="water_opt.traj")
    opt.run(fmax=0.01)  # Optimize until forces < 0.01 eV/Å

    print(f"Final energy: {water.get_potential_energy():.6f} eV")
    write("water_optimized.xyz", water)
