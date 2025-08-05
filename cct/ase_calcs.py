import contextlib
import copy
import os
from typing import Literal

import numpy as np
import pymsym
import torch
import wget
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from cct.gxtb import gxTBCalculator
from cct.rdkit.coordinates import set_coordinates
from cct.rdkit.io import to_ase
from cct.utils import PROJECT_ROOT, batch_run

QM_METHODS = Literal[
    "ANI1x",
    "ANI2x",
    "MACE_SMALL",
    "MACE_MEDIUM",
    "MACE_LARGE",
    "EGRET1",
    "GFN1-xTB",
    "GFN2-xTB",
    "gxTB",
]

EGRET1_URL = (
    "https://github.com/rowansci/egret-public/raw/refs/heads/master/"
    "compiled_models/EGRET_1.model"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class CalculatorFactory:
    """Toolkit for molecular energy calculations using various methods."""

    def __init__(self):
        """Initialize the energy toolkit."""
        self.egret1 = None

    def load_egret1(self):
        """Load the EGRET1 machine learning model for energy calculations."""
        if self.egret1 is None:
            from mace.calculators import mace_off

            egret1_path = PROJECT_ROOT / "data" / "EGRET_1.model"

            if not egret1_path.exists():
                print("Downloading EGRET1 model")

                wget.download(
                    EGRET1_URL,
                    out=str(egret1_path),
                )

            self.egret1 = mace_off(model=egret1_path, default_dtype="float32")
        return self.egret1

    def get_calc(self, method: QM_METHODS, solvent: Literal["water"] | None = None):
        """Get appropriate calculator for the specified quantum mechanical method.

        Args:
        ----
            method: Quantum mechanical method to use.
            solvent: Optional solvent for implicit solvation.

        Returns:
        -------
            ASE calculator instance.

        """
        match method:
            case "MACE_SMALL":
                from mace.calculators import mace_off

                calc = mace_off(model="small", device=device)
            case "MACE_MEDIUM":
                from mace.calculators import mace_off

                calc = mace_off(model="medium", device=device)
            case "MACE_LARGE":
                from mace.calculators import mace_off

                calc = mace_off(model="large", device=device)
            case "ANI1x":
                import torchani

                calc = torchani.models.ANI1x().ase()
            case "ANI2x":
                import torchani

                calc = torchani.models.ANI2x().ase()
            case "EGRET1":
                calc = self.load_egret1()
            case "gxTB":
                calc = gxTBCalculator()
            case "GFN1-xTB" | "GFN2-xTB":
                from tblite.ase import TBLite

                solvation_param = None
                if solvent:
                    solvation_param = ("alpb", solvent)
                calc = TBLite(method=method, verbosity=0, solvation=solvation_param)

        return calc

    def solvation_energy(
        self,
        mol: Chem.Mol,
        /,
        *,
        multiplicity: int = 1,
        charge: int = 0,
        solvent: Literal["water"] = "water",
    ):
        """Calculate solvation energy for a molecule.

        Args:
        ----
            mol: RDKit molecule.
            multiplicity: Spin multiplicity.
            charge: Molecular charge.
            solvent: Solvent for implicit solvation calculations.

        Returns:
        -------
            Solvation energy in eV.

        """
        E_solvated = self.singlepoint(
            mol, method="GFN2-xTB", solvent=solvent, multiplicity=multiplicity
        )
        E_gas = self.singlepoint(mol, method="GFN2-xTB", solvent=None, charge=charge)

        E_solv = E_solvated - E_gas

        return E_solv

    def singlepoint(
        self,
        mol: Chem.Mol,
        /,
        *,
        multiplicity: int = 1,
        charge: int = 0,
        method: QM_METHODS,
        solvent: Literal["water"] | None = None,
    ):
        """Calculate single-point energies for all conformers in a molecule.

        Args:
        ----
            mol: RDKit molecule with conformers.
            multiplicity: Spin multiplicity.
            charge: Molecular charge.
            method: Quantum mechanical method to use.
            solvent: Optional solvent for implicit solvation.

        Returns:
        -------
            Single energy (float) if one conformer, otherwise array of energies.

        """
        energies = []
        for conf_id in tqdm(range(mol.GetNumConformers())):
            energy = self._singlepoint(
                mol,
                conf_id=conf_id,
                method=method,
                solvent=solvent,
                multiplicity=multiplicity,
                charge=charge,
            )
            energies.append(energy)

        if len(energies) == 1:
            energies = energies[0]
        return np.array(energies)

    def _thermochemistry(
        self,
        mol: Chem.Mol,
        /,
        *,
        conf_id: int = -1,
        multiplicity: int = 1,
        charge: int = 0,
        method: QM_METHODS,
        solvent: Literal["water"] | None = None,
        temperature: float = 298.15,
        pressure: float = 101325.0,
    ):
        """Calculate single-point energy for a molecule."""
        atoms: Atoms = to_ase(mol, conf_id=conf_id)

        symmetry_number = pymsym.get_symmetry_number(
            atoms.get_atomic_numbers(), atoms.get_positions()
        )
        spin = (multiplicity - 1) / 2

        calc = self.get_calc(method=method, solvent=solvent)

        calc.set(charge=charge)
        calc.set(multiplicity=multiplicity)
        atoms.calc = calc

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            calc.calculate(atoms, properties=["energy"], system_changes=all_changes)
        potential_energy = calc.results["energy"]

        vib = Vibrations(atoms)
        vib.run()
        vib_energies = vib.get_energies()

        # Thermochemistry
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            potentialenergy=potential_energy,
            atoms=atoms,
            geometry="nonlinear",
            symmetrynumber=symmetry_number,
            spin=spin,
        )

        G = thermo.get_gibbs_energy(temperature=temperature, pressure=pressure)

        return G, vib_energies

    def _singlepoint(
        self,
        mol: Chem.Mol,
        /,
        *,
        conf_id: int = -1,
        multiplicity: int = 1,
        charge: int = 0,
        method: QM_METHODS,
        solvent: Literal["water"] | None = None,
    ):
        """Calculate single-point energy for a molecule."""
        atoms: Atoms = to_ase(mol, conf_id=conf_id)

        calc = self.get_calc(method=method, solvent=solvent)

        calc.set(charge=charge)
        calc.set(multiplicity=multiplicity)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            calc.calculate(atoms, properties=["energy"], system_changes=all_changes)
        energy = calc.results["energy"]

        mol.GetConformer(conf_id).SetProp("Energy", str(energy))

        return energy

    def optimise(
        self,
        mol: Chem.Mol,
        /,
        *,
        multiplicity: int = 1,
        charge: int = 0,
        method: QM_METHODS,
        max_iterations: int = 500,
        fmax: float = 0.05,
        solvent: Literal["water"] | None = None,
    ) -> Chem.Mol:
        """Optimize molecular geometries for all conformers.

        Args:
        ----
            mol: RDKit molecule with conformers.
            multiplicity: Spin multiplicity.
            charge: Molecular charge.
            method: Quantum mechanical method to use.
            max_iterations: Maximum optimization steps.
            fmax: Force convergence threshold.
            solvent: Optional solvent for implicit solvation.

        Returns:
        -------
            Tuple of optimized molecule and array of energies.

        """
        atoms_list = [
            to_ase(mol, conf_id=conf_id).copy()
            for conf_id in range(mol.GetNumConformers())
        ]
        calc = self.get_calc(method=method, solvent=solvent)
        calc.set(charge=charge)
        calc.set(multiplicity=multiplicity)

        runtime_env = {
            "num_cpus": 1,
            "runtime_env": {
                "env_vars": {
                    # force `ray` to not kill a proces on OOM but use SWAP instead
                    "RAY_DISABLE_MEMORY_MONITOR": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                }
            },
        }

        inputs = [(atoms, calc, max_iterations, fmax) for atoms in atoms_list]
        outputs = batch_run(
            ase_optimise, inputs, ray_kwargs=runtime_env, run_local=len(atoms_list) == 1
        )
        atoms_list, energies = zip(*outputs, strict=False)

        for conf_id, atoms in enumerate(atoms_list):
            set_coordinates(mol, atoms.get_positions(), conf_id=conf_id)

        return mol, np.array(energies)

    def _optimise(
        self,
        mol: Chem.Mol,
        /,
        *,
        conf_id: int = -1,
        multiplicity: int = 1,
        charge: int = 0,
        method: QM_METHODS = "EGRET1",
        max_iterations: int = 500,
        fmax: float = 0.05,
        solvent: Literal["water"] | None = None,
    ) -> Chem.Mol:
        """Optimize molecular geometry and return optimized molecule with energy."""
        atoms = to_ase(mol, conf_id=conf_id)

        calc = self.get_calc(method=method, solvent=solvent)
        calc.set(charge=charge)
        calc.set(multiplicity=multiplicity)

        atoms.calc = calc
        dyn = BFGS(atoms, logfile=None)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            dyn.run(fmax=fmax, steps=max_iterations)
        set_coordinates(mol, atoms.get_positions(), conf_id=conf_id)
        calc.calculate(atoms, properties=["energy"], system_changes=all_changes)
        energy = calc.results["energy"]
        mol.GetConformer(conf_id).SetProp("Energy", str(energy))

        return energy


def ase_optimise(
    atoms: Atoms, calc: Calculator, max_iterations: int = 500, fmax: float = 0.05
) -> tuple[Atoms, float]:
    """Optimize molecular geometry using ASE.

    Args:
    ----
        atoms: ASE Atoms object.
        calc: ASE calculator instance.
        max_iterations: Maximum optimization steps.
        fmax: Force convergence threshold.

    Returns:
    -------
        Tuple of optimized atoms and final energy.

    """
    atoms = (
        atoms.copy()
    )  # Copy to prevent the sharing of Atom objects between function calls
    calc = copy.deepcopy(calc)  # Same for the calculator
    atoms.calc = calc
    dyn = BFGS(atoms, logfile=None)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        dyn.run(fmax=fmax, steps=max_iterations)
    energy = atoms.get_potential_energy()
    return atoms, energy


if __name__ == "__main__":
    calcs = CalculatorFactory()
    mol = Chem.MolFromSmiles("O")
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)

    g = calcs._thermochemistry(mol, method="MACE_SMALL")
    print(g)

    # s=calcs._thermochemistry(mol, method="EGRET1", solvent="water")

    # print((s-g)*EV2KCALMOL)
