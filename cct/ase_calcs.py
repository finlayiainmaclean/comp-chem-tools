import contextlib
import multiprocessing
import os
import tempfile
import time
from typing import Literal

import numpy as np
import pymsym
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.nwchem import NWChem
from ase.optimize import LBFGS
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from rdkit import Chem
from rdkit.Chem import AllChem

from cct.gxtb import gxTBCalculator
from cct.rdkit.coordinates import set_coordinates
from cct.rdkit.io import ase_from_rdkit
from cct.utils import PROJECT_ROOT, batch_run, get_best_device, get_cuda_or_cpu_device

QM_METHODS = Literal[
    "ANI1x",
    "ANI2x",
    "GFN1-xTB",
    "GFN2-xTB",
    "gxTB",
    "UMA_MEDIUM",
    "UMA_SMALL",
    "EMT",
    "NWCHEM",
    # "MACE_SMALL",
    # "MACE_MEDIUM",
    # "MACE_LARGE",
    # "EGRET1",
]

# EGRET1_URL = "https://github.com/rowansci/egret-public/raw/refs/heads/master/compiled_models/EGRET_1.model"
RUN_LOCAL = eval(os.environ.get("RUN_LOCAL", "False"))
device = get_best_device()
cuda_or_cpu_device = get_cuda_or_cpu_device()
num_cpus = os.environ.get("NUM_CPUS", multiprocessing.cpu_count())


def get_calc(method: QM_METHODS, solvent: Literal["water"] | None = None, charge: int = 0, multiplicity: int = 1):
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
        case "EMT":
            from ase.calculators.emt import EMT

            calc = EMT()
        case "UMA_SMALL":
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            from fairchem.core.units.mlip_unit import load_predict_unit

            atom_refs = pretrained_mlip.get_isolated_atomic_energies("uma-s-1p1", "/tmp")
            predictor = load_predict_unit(
                PROJECT_ROOT / "models" / "uma-s-1p1.pt",
                inference_settings="default",
                overrides=None,
                device=cuda_or_cpu_device,
                atom_refs=atom_refs,
            )
            calc = FAIRChemCalculator(predictor, task_name="omol")
        case "UMA_MEDIUM":
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            from fairchem.core.units.mlip_unit import load_predict_unit

            atom_refs = pretrained_mlip.get_isolated_atomic_energies("uma-m-1p1", "/tmp")
            predictor = load_predict_unit(
                PROJECT_ROOT / "models" / "uma-m-1p1.pt",
                inference_settings="default",
                overrides=None,
                device=cuda_or_cpu_device,
                atom_refs=atom_refs,
            )
            calc = FAIRChemCalculator(predictor, task_name="omol")
        case "ANI1x":
            import torchani

            calc = torchani.models.ANI1x().ase()
        case "ANI2x":
            import torchani

            calc = torchani.models.ANI2x().ase()
        case "gxTB":
            calc = gxTBCalculator(charge=charge, multiplicity=multiplicity)
        case "GFN1-xTB" | "GFN2-xTB":
            from tblite.ase import TBLite

            solvation_param = None
            if solvent:
                solvation_param = ("alpb", solvent)
            calc = TBLite(
                method=method, verbosity=0, solvation=solvation_param, charge=charge, multiplicity=multiplicity
            )

        case "NWCHEM":
            os.environ["ASE_NWCHEM_COMMAND"] = f"mpirun -n {num_cpus} nwchem PREFIX.nwi > PREFIX.nwo"
            calc = NWChem(
                label="nwchem",
                dft={
                    "odft": None,
                    "mult": multiplicity,
                    "disp": "vdw 4",
                    "xc": "PBE0",
                    "convergence": {
                        "energy": 1e-9,
                        "density": 1e-7,
                        "gradient": 5e-6,
                    },
                },
                basis="def2-TZVP",
            )

            if solvent:
                calc.set(cosmo={"do_cosmo_smd": "true", "solvent": solvent})

        # case "AIMNet2":
        #     from aimnet.calculators import AIMNet2ASE
        #     calc = AIMNet2ASE()
        # case "eSEN-S":
        #     from fairchem.core import FAIRChemCalculator, pretrained_mlip

        #     predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-omol", device="cpu")
        #     calc = FAIRChemCalculator(predictor, task_name="omol")
        # case "MACE_SMALL":
        #     from mace.calculators import mace_off

        #     calc = mace_off(model="small", device="cpu", default_dtype="float32")
        #     calc.device = torch.device(device)
        #     calc.models = [model.to(device) for model in calc.models]
        # case "MACE_MEDIUM":
        #     from mace.calculators import mace_off
        #     calc = mace_off(model="medium", device="cpu", default_dtype="float32")
        #     calc.device = torch.device(device)
        #     calc.models = [model.to(device) for model in calc.models]
        # case "MACE_LARGE":
        #     from mace.calculators import mace_off
        #     calc = mace_off(model="large", device="cpu", default_dtype="float32")
        #     calc.device = torch.device(device)
        #     calc.models = [model.to(device) for model in calc.models]
        # case "EGRET1":
        #     from mace.calculators import mace_off

        #     egret1_path = PROJECT_ROOT / "data" / "EGRET_1.model"

        #     if not egret1_path.exists():
        #         print("Downloading EGRET1 model")

        #         wget.download(
        #             EGRET1_URL,
        #             out=str(egret1_path),
        #         )

        #     calc = mace_off(model=egret1_path, device="cpu", default_dtype="float32")
        #     calc.device = torch.device(device)
        #     calc.models = [model.to(device) for model in calc.models]
    return calc


def ase_optimise(
    atoms: Atoms,
    calc: QM_METHODS,
    charge: int = 0,
    multiplicity: int = 1,
    solvent: Literal["water"] | None = None,
    max_iterations: int = 5000,
    fmax: float = 0.05,
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
    atoms = atoms.copy()  # Copy to prevent the sharing of Atom objects between function calls
    if not isinstance(calc, Calculator):
        calc = get_calc(method=calc, solvent=solvent, multiplicity=multiplicity, charge=charge)
    else:
        calc = calc.copy()

    atoms.info.update({"charge": charge})
    atoms.info.update({"multiplicity": multiplicity})
    calc.set(charge=charge)
    if not isinstance(calc, NWChem):
        spin = (multiplicity - 1) // 2
        atoms.info.update({"multiplicity": multiplicity, "spin": spin})
        calc.set(multiplicity=multiplicity)
        calc.set(spin=spin)

    if hasattr(calc, "set_charge") and callable(calc.set_charge):
        calc.set_charge(charge)
    if hasattr(calc, "set_atoms") and callable(calc.set_atoms):
        calc.set_atoms(atoms)
    if hasattr(calc, "set_mult") and callable(calc.set_mult):
        calc.set_mult(multiplicity)
    else:
        atoms.calc = calc

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        dyn = LBFGS(atoms, logfile=None)
        dyn.run(fmax=fmax, steps=max_iterations)

    energy = atoms.get_potential_energy()
    atoms.calc = None

    return atoms, energy


def ase_singlepoint(
    atoms: Atoms,
    calc: QM_METHODS | Calculator,
    charge: int = 0,
    multiplicity: int = 1,
    solvent: Literal["water"] | None = None,
) -> float:
    """Optimize molecular geometry using ASE.

    Args:
    ----
        atoms: ASE Atoms object.
        method: ASE calculator.

    Returns:
    -------
        Electronic energy.

    """

    atoms = atoms.copy()  # Copy to prevent the sharing of Atom objects between function calls
    if not isinstance(calc, Calculator):
        calc = get_calc(method=calc, solvent=solvent, charge=charge, multiplicity=multiplicity)
    else:
        calc = calc.copy()

    atoms.info.update({"charge": charge})
    calc.set(charge=charge)
    if not isinstance(calc, NWChem):
        spin = (multiplicity - 1) // 2
        atoms.info.update({"multiplicity": multiplicity, "spin": spin})
        calc.set(multiplicity=multiplicity)
        calc.set(spin=spin)

    atoms.calc = calc

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        calc.calculate(atoms, properties=["energy"], system_changes=all_changes)
        energy = calc.results["energy"]
    return energy


class CalculatorFactory:
    """Toolkit for molecular energy calculations using various methods."""

    def __init__(self):
        """Initialize the energy toolkit."""
        self.egret1 = None

    def solvation_energy_mol(
        self,
        mol: Chem.Mol,
        /,
        *,
        multiplicity: int = 1,
        method: Literal["GFN2-xTB"] = "GFN2-xTB",
        charge: int | None = None,
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
        if not charge:
            charge = Chem.GetFormalCharge(mol)
        match method:
            case "GFN2-xTB":
                E_solvated = self.singlepoint_mol(mol, method="GFN2-xTB", solvent=solvent, multiplicity=multiplicity)
                E_gas = self.singlepoint_mol(
                    mol, method="GFN2-xTB", solvent=None, charge=charge, multiplicity=multiplicity
                )
                E_solv = E_solvated - E_gas
        return E_solv

    def singlepoint_mol(
        self,
        mol: Chem.Mol,
        /,
        *,
        charge: int | None = None,
        multiplicity: int = 1,
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
        if not charge:
            charge = Chem.GetFormalCharge(mol)

        atoms_list = [ase_from_rdkit(mol, conf_id=conf_id).copy() for conf_id in range(mol.GetNumConformers())]

        runtime_env = {
            "num_cpus": 1,
            "runtime_env": {
                "env_vars": {
                    # force `ray` to not kill a proces on OOM but use SWAP instead
                    "RAY_DISABLE_MEMORY_MONITOR": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OMP_MAX_ACTIVE_LEVELS": "1",
                    "OMP_STACKSIZE": "4G",
                }
            },
        }

        inputs = [(atoms, method, charge, multiplicity, solvent) for atoms in atoms_list]
        energies = batch_run(
            ase_singlepoint, inputs, ray_kwargs=runtime_env, run_local=len(atoms_list) == 1 or RUN_LOCAL
        )

        return np.array(energies)

    def thermochemistry_conf(
        self,
        mol: Chem.Mol,
        /,
        *,
        conf_id: int = -1,
        multiplicity: int = 1,
        charge: int | None = None,
        method: QM_METHODS,
        solvent: Literal["water"] | None = None,
        temperature: float = 298.15,
        pressure: float = 101325.0,
    ):
        """Calculate single-point energy for a molecule."""
        if not charge:
            charge = Chem.GetFormalCharge(mol)
        atoms: Atoms = ase_from_rdkit(mol, conf_id=conf_id)

        symmetry_number = pymsym.get_symmetry_number(atoms.get_atomic_numbers(), atoms.get_positions())

        calc = get_calc(method=method, solvent=solvent, charge=charge, multiplicity=multiplicity)

        atoms.calc = calc

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            calc.calculate(atoms, properties=["energy"], system_changes=all_changes)
            potential_energy = calc.results["energy"]

            with tempfile.TemporaryDirectory() as tmpdir:  # Wrap in tmp to prevent caching between mols
                vib = Vibrations(atoms, nfree=2, name=tmpdir)
                vib.run()  # Wrap in devnull to prevent the printing of the summary
                vib_energies = vib.get_energies(read_cache=False)

        # Thermochemistry
        spin = (multiplicity - 1) / 2
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            potentialenergy=potential_energy,
            atoms=atoms,
            geometry="nonlinear",
            symmetrynumber=symmetry_number,
            spin=spin,
            ignore_imag_modes=True,
        )

        G = thermo.get_gibbs_energy(temperature=temperature, pressure=pressure)

        return G, vib_energies

    def singlepoint_conf(
        self,
        mol: Chem.Mol,
        /,
        *,
        conf_id: int = 0,
        multiplicity: int = 1,
        charge: int | None = None,
        method: QM_METHODS,
        solvent: Literal["water"] | None = None,
    ):
        """Calculate single-point energy for a molecule."""
        if not charge:
            charge = Chem.GetFormalCharge(mol)
        atoms: Atoms = ase_from_rdkit(mol, conf_id=conf_id)

        if not charge:
            charge = Chem.GetFormalCharge(mol)

        calc = get_calc(method=method, solvent=solvent, charge=charge, multiplicity=multiplicity)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            calc.calculate(atoms, properties=["energy"], system_changes=all_changes)
        energy = calc.results["energy"]

        mol.GetConformer(conf_id).SetProp("Energy", str(energy))

        return energy

    def interaction_conf(
        self,
        /,
        *,
        molA: Chem.Mol,
        molB: Chem.Mol,
        molAB: Chem.Mol,
        conf_idA: int = 0,
        conf_idB: int = 0,
        conf_idAB: int = 0,
        multiplicityA: int = 1,
        multiplicityB: int = 1,
        multiplicityAB: int = 1,
        chargeA: int | None = None,
        chargeB: int | None = None,
        chargeAB: int | None = None,
        method: QM_METHODS,
        solvent: Literal["water"] | None = None,
    ):
        energyA = self.singlepoint_conf(
            molA, conf_id=conf_idA, charge=chargeA, multiplicity=multiplicityA, method=method, solvent=solvent
        )
        energyB = self.singlepoint_conf(
            molB, conf_id=conf_idB, charge=chargeB, multiplicity=multiplicityB, method=method, solvent=solvent
        )
        energyAB = self.singlepoint_conf(
            molAB, conf_id=conf_idAB, charge=chargeAB, multiplicity=multiplicityAB, method=method, solvent=solvent
        )

        interaction_energy = energyAB - (energyA + energyB)
        return interaction_energy

    def optimise_mols(
        self,
        mols: list[Chem.Mol],
        /,
        *,
        multiplicity: list[int] | None = None,
        charge: list[int] | None = None,
        method: QM_METHODS,
        max_iterations: int = 500,
        fmax: float = 0.01,
        solvent: Literal["water"] | None = None,
    ) -> tuple[list[Chem.Mol], np.ndarray[float]]:
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
        if not multiplicity:
            multiplicity = [1] * len(mols)
        if not charge:
            charge = [0] * len(mols)

        inputs = []
        for mol in mols:
            atoms = ase_from_rdkit(mol, conf_id=-1).copy()
            inputs.append((atoms, method, charge, multiplicity, solvent, max_iterations, fmax))

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

        outputs = batch_run(ase_optimise, inputs, ray_kwargs=runtime_env, run_local=len(inputs) == 1 or RUN_LOCAL)
        atoms_list, energies = zip(*outputs, strict=False)

        for mol, atoms in zip(mols, atoms_list):
            set_coordinates(mol, atoms.get_positions(), conf_id=-1)
        return mols, np.array(energies)

    def optimise_mol(
        self,
        mol: Chem.Mol,
        /,
        *,
        multiplicity: int = 1,
        charge: int | None = None,
        method: QM_METHODS,
        max_iterations: int = 500,
        fmax: float = 0.01,
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
        if not charge:
            charge = Chem.GetFormalCharge(mol)
        atoms_list = [ase_from_rdkit(mol, conf_id=conf_id).copy() for conf_id in range(mol.GetNumConformers())]

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

        inputs = [(atoms, method, charge, multiplicity, solvent, max_iterations, fmax) for atoms in atoms_list]
        outputs = batch_run(ase_optimise, inputs, ray_kwargs=runtime_env, run_local=len(atoms_list) == 1 or RUN_LOCAL)
        atoms_list, energies = zip(*outputs, strict=False)

        for conf_id, atoms in enumerate(atoms_list):
            set_coordinates(mol, atoms.get_positions(), conf_id=conf_id)

        return mol, np.array(energies)

    def optimise_conf(
        self,
        mol: Chem.Mol,
        /,
        *,
        conf_id: int = -1,
        multiplicity: int = 1,
        charge: int | None = None,
        method: QM_METHODS = "EGRET1",
        max_iterations: int = 5000,
        fmax: float = 0.05,
        solvent: Literal["water"] | None = None,
    ) -> Chem.Mol:
        """Optimize molecular geometry and return optimized molecule with energy."""
        if not charge:
            charge = Chem.GetFormalCharge(mol)

        atoms = ase_from_rdkit(mol, conf_id=conf_id)

        calc = get_calc(method=method, solvent=solvent, charge=charge, multiplicity=multiplicity)

        atoms.calc = calc
        dyn = LBFGS(atoms, logfile=None)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            dyn.run(fmax=fmax, steps=max_iterations)
            set_coordinates(mol, atoms.get_positions(), conf_id=conf_id)
            calc.calculate(atoms, properties=["energy"], system_changes=all_changes)
        energy = calc.results["energy"]
        mol.GetConformer(conf_id).SetProp("Energy", str(energy))

        return energy


if __name__ == "__main__":
    calcs = CalculatorFactory()
    mol = Chem.MolFromSmiles("O")
    mol = Chem.AddHs(mol)

    AllChem.EmbedMolecule(mol)

    t0 = time.time()
    g = calcs.singlepoint_conf(mol, method="NWCHEM__PBE0__def2-TZVP")

    gs = calcs.singlepoint_conf(mol, method="NWCHEM__PBE0__def2-TZVP", solvent="water")
    print(gs - g)
    print(time.time() - t0)

    # g = calcs.solvation_energy_mol(mol, method="GFN2-xTB", solvent="water")*23

    # print(g,g1,g-g1)

    # # s=calcs._thermochemistry(mol, method="EGRET1", solvent="water")

    # # print((s-g)*EV2KCALMOL)
