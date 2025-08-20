import logging
import math
import os
import sys
import warnings
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from datamol.mol import same_mol
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen
from torch.utils.data import DataLoader

from cct.pka.conformer import ConformerGen
from cct.pka.dataset import MolDataset
from cct.pka.model import UniMolModel
from cct.pka.template import LN10, TRANSLATE_PH, enumerate_template, get_ensemble, log_sum_exp, prot, read_template
from cct.rdkit.coordinates import transplant_coordinates
from cct.utils import PROJECT_ROOT

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings(action="ignore")
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unimol_free_energy.inference")
logging.disable(50)


class UnipKa(object):
    def __init__(self, batch_size=32, remove_hs=False, use_gpu=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        model_path = PROJECT_ROOT / "models" / "uni-pka-ckpt_v2" / "t_dwar_v_novartis_a_b.pt"

        pattern_path = PROJECT_ROOT / "models" / "uni-pka-ckpt_v2" / "simple_smarts_pattern.tsv"

        self.model = UniMolModel(model_path, output_dim=1, remove_hs=remove_hs).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.params = {"remove_hs": remove_hs}
        self.conformer_gen = ConformerGen(**self.params)
        self.template_a2b, self.template_b2a = read_template(pattern_path)

    #### Internal functions ####

    @staticmethod
    def _get_formal_charge(mol):
        """
        Calculate the sum of formal charges on all atoms in the molecule.
        This represents the total formal charge of the microstate.
        """
        if mol is None:
            return float("inf")  # Invalid molecule

        formal_charges = []
        for atom in mol.GetAtoms():
            formal_charges.append(atom.GetFormalCharge())

        abs_formal_charge = np.abs(np.sum(formal_charges))
        abs_atoms_charges = np.sum([abs(charge) for charge in formal_charges])
        return abs_formal_charge, abs_atoms_charges

    def _preprocess_data(self, smiles_list):
        inputs = self.conformer_gen.transform(smiles_list)
        return inputs

    def _predict(self, smiles: list[str] | str):
        if isinstance(smiles, str):
            smiles = [smiles]
        unimol_input = self._preprocess_data(smiles)
        dataset = MolDataset(unimol_input)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.model.batch_collate_fn,
        )

        results = {}
        for batch in dataloader:
            net_input, _ = self._decorate_torch_batch(batch)
            with torch.no_grad():
                predictions = self.model(**net_input)
                for smiles, energy in zip(smiles, predictions):
                    results[smiles] = energy.item()
        return results

    def _decorate_torch_batch(self, batch):
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {"net_input": net_input.to(self.device)}, net_target.to(self.device)
        net_target = None

        return net_input, net_target

    def _predict_micro_pKa(self, smi: str, idx: int, mode: Literal["a2b", "b2a"]):
        mol = Chem.MolFromSmiles(smi)
        new_mol = Chem.RemoveHs(prot(mol, idx, mode))
        new_smi = Chem.MolToSmiles(new_mol)
        if mode == "a2b":
            smi_A = smi
            smi_B = new_smi
        elif mode == "b2a":
            smi_B = smi
            smi_A = new_smi
        DfGm = self._predict([smi_A, smi_B])
        pKa = (DfGm[smi_B] - DfGm[smi_A]) / LN10 + TRANSLATE_PH
        return pKa

    def _predict_macro_pKa(self, smi: str, mode: Literal["a2b", "b2a"]) -> float:
        macrostate_A, macrostate_B = enumerate_template(smi, self.template_a2b, self.template_b2a, mode)
        DfGm_A = self._predict(macrostate_A)
        DfGm_B = self._predict(macrostate_B)
        return log_sum_exp(DfGm_A.values()) - log_sum_exp(DfGm_B.values()) + TRANSLATE_PH

    def _predict_macro_pKa_from_macrostate(self, macrostate_A, macrostate_B) -> float:
        DfGm_A = self._predict(macrostate_A)
        DfGm_B = self._predict(macrostate_B)
        return log_sum_exp(DfGm_A.values()) - log_sum_exp(DfGm_B.values()) + TRANSLATE_PH

    def _predict_ensemble_free_energy(self, smi: str) -> Dict[int, Tuple[str, float]]:
        ensemble = get_ensemble(smi, self.template_a2b, self.template_b2a)
        ensemble_free_energy = dict()
        for q, macrostate in ensemble.items():
            prediction = self._predict(macrostate)
            _ensemble_free_energy = []
            for microstate in macrostate:
                if microstate in prediction:
                    _ensemble_free_energy.append((microstate, prediction[microstate]))
            ensemble_free_energy[q] = _ensemble_free_energy

        if len(ensemble_free_energy) == 0:
            raise ValueError("Could not process any microstates")
        return ensemble_free_energy

    #### Public functions ####

    def get_acidic_macro_pka(self, smi: str) -> float:
        return self._predict_macro_pKa(smi, mode="a2b")

    def get_basic_macro_pka(self, smi: str) -> float:
        return self._predict_macro_pKa(smi, mode="b2a")

    def get_acidic_micro_pka(self, smi: str, idx: int) -> float:
        return self._predict_micro_pKa(smi, mode="a2b", idx=idx)

    def get_basic_micro_pka(self, smi: str, idx: int) -> float:
        return self._predict_micro_pKa(smi, mode="b2a", idx=idx)

    def get_distribution(self, mol: Chem.Mol | str, pH: float = 7.4) -> pd.DataFrame:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)

        query_smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))

        # Free energy predictions from your model, grouped by charge
        ensemble_free_energy = self._predict_ensemble_free_energy(query_smi)

        records = []
        partition_function = 0.0

        # Collect Boltzmann weights and energy terms
        for q, macrostate_free_energy in ensemble_free_energy.items():
            for microstate_smi, DfGm in macrostate_free_energy:
                G_pH = DfGm + q * LN10 * (pH - TRANSLATE_PH)  # pH-adjusted free energy
                boltzmann_factor = math.exp(-G_pH)
                records.append((q, microstate_smi, DfGm, G_pH, boltzmann_factor))
                partition_function += boltzmann_factor

        # Normalize to get population w_i(pH)
        df = pd.DataFrame(
            records, columns=["charge", "smiles", "free_energy", "adjusted_free_energy", "boltzmann_factor"]
        )
        df["population"] = df["boltzmann_factor"] / partition_function

        # Sort for readability
        df = df.sort_values(by="population", ascending=False).reset_index(drop=True)

        # Optional: add mol objects and coordinate mapping
        df["mol"] = df.smiles.apply(Chem.MolFromSmiles)

        if mol.GetNumConformers() > 0:
            df["mol"] = df.mol.apply(lambda x: transplant_coordinates(mol, x))  # if you have this
        df["is_query_mol"] = df["mol"].apply(lambda x: same_mol(mol, x, use_non_standard_inchikey=True))

        return df

    def get_state_penalty(self, mol: Chem.Mol | str, T: float = 298.15, pH: float = 7.4) -> float:
        """
        Calculate the state penalty (SP) according to the Lawrenz concept.

        Selects formally neutral microstates that minimize atom-centered charges,
        preferring non-zwitterionic forms over zwitterionic counterparts.
        """
        R = 8.314  # J/mol/K
        df = self.get_distribution(mol, pH=pH)

        # Calculate formal charges for all molecules
        charge_results = df["mol"].apply(self._get_formal_charge)
        df["abs_formal_charge"] = [result[0] for result in charge_results]
        df["abs_atoms_charges"] = [result[1] for result in charge_results]

        # Step 1: Find microstates with minimum absolute formal charge (preferably 0)
        min_abs_formal_charge = df["abs_formal_charge"].min()
        neutral_candidates = df[df["abs_formal_charge"] == min_abs_formal_charge].copy()

        if min_abs_formal_charge == 0:
            # Step 2: Among neutral microstates, prefer those with minimum atom-centered charges
            # This favors non-zwitterionic forms over zwitterionic forms
            min_atom_charges = neutral_candidates["abs_atoms_charges"].min()
            reference_microstates_df = neutral_candidates[
                neutral_candidates["abs_atoms_charges"] == min_atom_charges
            ].copy()
        else:
            # No truly neutral forms exist - use microstates with minimum formal charge
            # Among these, still prefer those with minimum atom-centered charges
            min_atom_charges = neutral_candidates["abs_atoms_charges"].min()
            reference_microstates_df = neutral_candidates[
                neutral_candidates["abs_atoms_charges"] == min_atom_charges
            ].copy()

        # Sort reference microstates by population for inspection
        reference_microstates_df = reference_microstates_df.sort_values(by="population", ascending=False).reset_index(
            drop=True
        )

        # Calculate sum of reference microstate populations
        sum_reference_pop = reference_microstates_df["population"].sum()

        if sum_reference_pop <= 0:
            raise ValueError("Error: No population in reference microstates!")

        if sum_reference_pop < 1e-10:
            raise ValueError(
                f"Warning: Very low reference population ({sum_reference_pop:.2e}). State penalty may be unreliable."
            )

        # Calculate state penalty: SP = -RT * ln(sum of reference populations)
        SP_J_mol = -R * T * math.log(sum_reference_pop)
        SP_kcal_mol = SP_J_mol / (4.184 * 1000)  # Convert to kcal/mol

        return SP_kcal_mol, reference_microstates_df

    def get_logd(self, mol: Chem.Mol | str, pH: float) -> float:
        """
        Compute logD(pH) from microstate populations and logP values.

        Parameters:
        - df: DataFrame output from compute_microstate_populations_at_pH, must contain:
            - 'mol': RDKit Mol object
            - 'charge': formal charge
            - 'population': w_i(pH)

        Returns:
        - logD (float): pH-dependent distribution coefficient
        """

        df = self.get_distribution(mol, pH=pH)

        logP_list = []
        weighted_linear_logP = []

        for _, row in df.iterrows():
            mol = row["mol"]
            charge = row["charge"]
            pop = row["population"]

            # logP for neutral species
            if charge == 0:
                logP = Crippen.MolLogP(mol)
            else:
                logP = -2.0  # fixed logP for ionic species

            logP_list.append(logP)
            weighted_linear_logP.append(pop * (10**logP))

        # Compute logD from weighted sum in linear space
        logd = np.log10(sum(weighted_linear_logP))

        # Optional: include in the DataFrame if you want to return it
        df["logP"] = logP_list
        df["weighted_linear_logP"] = weighted_linear_logP

        return logd
