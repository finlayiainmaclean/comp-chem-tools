import contextlib
import os
import tempfile

import torch
from ase.calculators.calculator import Calculator, all_changes
from datamol import to_sdf
from openbabel import pybel

from cct.consts import EV2KCALMOL
from cct.molsolv.descriptor import mol2vec
from cct.molsolv.model import load_model
from cct.rdkit.connectivity import determine_bonds
from cct.rdkit.io import rdkit_from_ase


class MolSolvCalculator(Calculator):
    """
    ASE Calculator for molecular solvation energy prediction using CCT models.

    This calculator computes solvation energies but does not compute forces
    or stress tensors as it's based on a molecular descriptor approach.
    """

    implemented_properties = ["energy"]
    default_parameters = {"charge": 0}

    def __init__(self, **kwargs):
        """
        Initialize the solvation calculator.

        Parameters:
        -----------
        charge : float, optional
            Molecular charge (default: 0 for neutral molecules)
        **kwargs : dict
            Additional arguments passed to the parent Calculator class
        """
        super().__init__(**kwargs)

        # Load the models once during initialization
        self.nmodel, self.imodel = load_model()
        self.device = "cpu"

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Calculate solvation energy for the given atoms object.

        Parameters:
        -----------
        atoms : ase.Atoms
            The atoms object to calculate solvation energy for
        properties : list
            List of properties to calculate (only 'energy' is supported)
        system_changes : list
            List of changes since last calculation
        """
        # Call parent calculate method
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        # Convert ASE atoms to OpenBabel molecule
        pmol = self._atoms_to_pybel(atoms)

        # Get charge from calculator parameters (defaults to 0 if not set)
        charge = self.parameters.get("charge", 0)

        pmol.OBMol.SetTotalCharge(int(charge))

        # Predict solvation energy
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            if abs(charge) > 0.0:
                solv_energy = self._predict(pmol, self.imodel)
            else:
                solv_energy = self._predict(pmol, self.nmodel)

        # Store results
        self.results["energy"] = solv_energy

    def _atoms_to_pybel(self, atoms):
        """
        Convert ASE atoms object to pybel molecule.

        Parameters:
        -----------
        atoms : ase.Atoms
            ASE atoms object

        Returns:
        --------
        pybel.Molecule
            OpenBabel molecule object
        """
        # Write atoms to temporary XYZ file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf") as tmp_file:
            charge = self.parameters.get("charge", 0)

            rdmol = rdkit_from_ase(atoms, charge=charge)
            determine_bonds(rdmol, charge=charge)
            to_sdf([rdmol], tmp_file.name)
            pmol = next(pybel.readfile("sdf", tmp_file.name))
            try:
                pmol.OBMol.PerceiveBondOrders()  # Force bond perception
                pmol.OBMol.SetAromaticPerceived()  # Mark aromatics as perceived
            except Exception as e:
                print(repr(e))
            return pmol

    def _predict(self, pmol, model):
        """
        Predict solvation energy for a pybel molecule.

        Parameters:
        -----------
        pmol : pybel.Molecule
            OpenBabel molecule object
        model : torch.nn.Module
            Trained solvation model

        Returns:
        --------
        float
            Predicted solvation energy
        """
        obmol = pmol.OBMol
        data = mol2vec(obmol)

        with torch.no_grad():
            data = data.to(self.device)
            solv = model(data).cpu().numpy()[0][0] / EV2KCALMOL

        return solv


# Example usage
if __name__ == "__main__":
    from ase.build import molecule

    # Create a water molecule using ASE
    water = molecule("H2O")

    # Create calculator (neutral by default)
    calc = MolSolvCalculator()

    # Attach calculator to atoms
    water.calc = calc

    # Get solvation energy (will use neutral model)
    energy = water.get_potential_energy()
    print(f"Solvation energy (neutral): {energy}")

    # Example with charged molecule
    calc_charged = MolSolvCalculator(charge=1)  # or use calc.set(charge=1)
    water.calc = calc_charged
    energy_charged = water.get_potential_energy()
    print(f"Solvation energy (charged): {energy_charged}")

    # Alternative way to set charge
    calc.set(charge=-1)
    water.calc = calc
    energy_anionic = water.get_potential_energy()
    print(f"Solvation energy (anionic): {energy_anionic}")
