from multiprocessing import cpu_count

import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from cct.ase_calcs import CalculatorFactory
from cct.consts import EV2KCALMOL
from cct.rdkit.conformers import embed, keep_conformers, sort_conformers_by_values
from cct.utils import boltzmann_average_energy
from cct.workflows.conformers import _generate_conformers_rdkit

initial_energy_threshold = 5.0
max_conformers_qm = 5
qm_method = "EGRET1"

calcs = CalculatorFactory()
enumerator = rdMolStandardize.TautomerEnumerator()
num_cores = cpu_count()

mol = Chem.MolFromSmiles("Nc1nnc[nH]1")
tauts = enumerator.Enumerate(mol)
tauts = list(tauts.tautomers)
charge = Chem.GetFormalCharge(tauts[0])

multiplicity = 1
optimised_tauts = []
energies = []
for taut in tauts:
    taut = Chem.AddHs(taut, addCoords=True)
    taut = embed(taut, num_confs=1)

    calcs.optimise_conf(taut, method="GFN2-xTB", multiplicity=multiplicity, charge=charge)
    gas_energy = calcs.singlepoint_conf(taut, method=qm_method, multiplicity=multiplicity, charge=charge)
    solvation_energy = calcs.solvation_energy_mol(taut, solvent="water", multiplicity=multiplicity, charge=charge)
    energy = gas_energy + solvation_energy
    energies.append(energy)
    optimised_tauts.append(taut)

energies = np.array(energies) * EV2KCALMOL
relative_energies = energies - np.min(energies)

low_energy_mask = relative_energies < initial_energy_threshold
low_energy_tauts = np.array(optimised_tauts)[low_energy_mask]

print(f"{len(low_energy_tauts)} low energy tautomers")


results = []
for taut in low_energy_tauts:
    smi = Chem.MolToSmiles(taut)
    taut = _generate_conformers_rdkit(
        taut,
        mode="rapid",
        multiplicity=multiplicity,
        charge=charge,
    )

    # Single point calculations
    print(f"Running QM singlepoint with {taut.GetNumConformers()} conformers")

    energies = calcs.singlepoint_mol(taut, method=qm_method, multiplicity=multiplicity, charge=charge)
    conf_ids_to_keep = np.argsort(energies)[:max_conformers_qm]
    keep_conformers(taut, conf_ids_to_keep)

    print(f"Running QM optimisation with {taut.GetNumConformers()} conformers")

    # Optimization
    taut, gas_phase_energies = calcs.optimise_mol(taut, method=qm_method, multiplicity=multiplicity, charge=charge)

    # Solvation energy
    solvation_energies = calcs.solvation_energy_mol(taut, multiplicity=multiplicity, charge=charge)

    # Combined energies
    solvated_energies = gas_phase_energies + solvation_energies
    taut, solvated_energies = sort_conformers_by_values(taut, solvated_energies)

    avg_solvated_energy = boltzmann_average_energy(solvated_energies)

    results.append({"mol": taut, "smi": smi, "avg_solvated_energy": avg_solvated_energy})

print(results)
