from typing import Literal

import wget
from ase import Atoms
from ase.calculators.calculator import all_changes
from ase.optimize import BFGS
from mace.calculators import mace_off
from rdkit import Chem
from rdkit.Chem import AllChem

from cct.energy.consts import EV2KCALMOL
from cct.rdkit.ase import ase_to_rdkit, rdkit_to_ase
from cct.utils import PROJECT_ROOT

METHODS = Literal["EGRET1"]

EGRET1_URL = (
    "https://github.com/rowansci/egret-public/raw/refs/heads/master/"
    "compiled_models/EGRET_1.model"
)


class EnergyToolkit:
    def __init__(self):
        self.egret1 = None

    def load_egret1(self):

        if self.egret1 is None:
            egret1_path = PROJECT_ROOT / "data" / "EGRET_1.model"
            print(egret1_path, egret1_path.exists())

            if not egret1_path.exists():
                print("Downloading EGRET1 model")

                wget.download(
                    EGRET1_URL,
                    out=str(egret1_path),
                )

            self.egret1 = mace_off(model=egret1_path, default_dtype="float64")
        return self.egret1

    def singlepoint(self, rdmol: Chem.Mol, method: METHODS = "EGRET1"):
        atoms: Atoms = rdkit_to_ase(rdmol)

        match method:
            case "EGRET1":
                calc = self.load_egret1()
                calc.calculate(atoms, ["energy"], all_changes)
                energy = calc.results["energy"] * EV2KCALMOL

        return energy

    def optimise(
        self,
        rdmol: Chem.Mol,
        method: METHODS = "EGRET1",
        max_iterations: int = 500,
        fmax: float = 0.05,
    ) -> Chem.Mol:

        atoms = rdkit_to_ase(rdmol)

        match method:
            case "EGRET1":
                calc = self.load_egret1()
                atoms.set_calculator(calc)
                dyn = BFGS(atoms, logfile=None)
                dyn.run(fmax=fmax, steps=max_iterations)
                mol_opt = ase_to_rdkit(atoms, rdmol)
                calc.calculate(atoms, ["energy"], all_changes)
                energy_opt = calc.results["energy"] * EV2KCALMOL

        return mol_opt, energy_opt


if __name__ == "__main__":
    etk = EnergyToolkit()
    mol = Chem.MolFromSmiles("CC")
    AllChem.EmbedMolecule(mol)

    print(etk.singlepoint(mol))
    print(etk.optimise(mol))
