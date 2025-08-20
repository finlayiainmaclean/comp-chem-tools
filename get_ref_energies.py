import numpy as np
import psi4
from scipy.optimize import minimize_scalar

# Setup Psi4
psi4.core.set_output_file("psi4_ccsdt_cbs_atoms.out", False)
psi4.set_memory("4 GB")
psi4.set_num_threads(4)

psi4.set_options({"reference": "uhf", "scf_type": "df", "e_convergence": 1e-8, "d_convergence": 1e-8})

# Basis sets and cardinal numbers
basis_list = ["def2-TZVPP", "def2-QZVPP", "def2-QZVPPD"]
cardinal = [3, 4, 4.5]  # QZVPPD is between QZ and 5Z quality

# Define atoms with their multiplicities
atoms = {
    "H": {"multiplicity": 2},  # doublet
    "Cl": {"multiplicity": 2},  # doublet
    "F": {"multiplicity": 2},  # doublet
    "Br": {"multiplicity": 2},  # doublet
    "I": {"multiplicity": 2},  # doublet
}


def calculate_cbs_energy(atom_symbol, multiplicity):
    """Calculate CBS energy for a single atom"""
    results = []

    print(f"\nCalculating {atom_symbol} atom (2S+1 = {multiplicity})...")
    print("Running calculations with 3 basis sets...")

    for i, (bas, L) in enumerate(zip(basis_list, cardinal)):
        print(f"  Calculation {i + 1}/3: {bas} (L={L})")

        # Clean up previous calculations
        psi4.core.clean()

        # Create fresh molecule object for each calculation
        mol = psi4.geometry(f"""
        0 {multiplicity}
        {atom_symbol} 0.0 0.0 0.0
        units angstrom
        no_reorient
        no_com
        """)

        # Set the basis set
        psi4.set_options({"basis": bas})

        escf = psi4.energy("scf", molecule=mol)
        ecc = psi4.energy("ccsd(t)", molecule=mol)
        ecorr = ecc - escf
        results.append({"L": L, "E_scf": escf, "E_corr": ecorr})

        print(f"    E_SCF = {escf:.8f} Ha")
        print(f"    E_CCSD(T) = {ecc:.8f} Ha")
        print(f"    E_corr = {ecorr:.8f} Ha")

    # Extract data for extrapolation
    L_vals = np.array([r["L"] for r in results])
    E_scf_vals = np.array([r["E_scf"] for r in results])
    E_corr_vals = np.array([r["E_corr"] for r in results])

    # Function to fit HF energy: E(L) = E_inf + A * exp(-alpha * L)
    def fit_hf_energy(alpha):
        """Fit HF energy with exponential form and return sum of squared residuals"""
        X = np.column_stack([np.ones(3), np.exp(-alpha * L_vals)])
        coeffs = np.linalg.lstsq(X, E_scf_vals, rcond=None)[0]
        E_inf, A = coeffs
        residuals = E_scf_vals - (E_inf + A * np.exp(-alpha * L_vals))
        return np.sum(residuals**2), E_inf

    # Function to fit correlation energy: E_corr(L) = E_inf + B * L^(-beta)
    def fit_corr_energy(beta):
        """Fit correlation energy with power law and return sum of squared residuals"""
        X = np.column_stack([np.ones(3), L_vals ** (-beta)])
        coeffs = np.linalg.lstsq(X, E_corr_vals, rcond=None)[0]
        E_inf, B = coeffs
        residuals = E_corr_vals - (E_inf + B * L_vals ** (-beta))
        return np.sum(residuals**2), E_inf

    # Optimize extrapolations
    res_hf = minimize_scalar(lambda alpha: fit_hf_energy(alpha)[0], bounds=(1.0, 10.0), method="bounded")
    alpha_hf_opt = res_hf.x
    _, E_HF_inf = fit_hf_energy(alpha_hf_opt)

    res_corr = minimize_scalar(lambda beta: fit_corr_energy(beta)[0], bounds=(1.5, 5.0), method="bounded")
    beta_corr_opt = res_corr.x
    _, E_corr_inf = fit_corr_energy(beta_corr_opt)

    # Final CBS energy
    E_CBS = E_HF_inf + E_corr_inf

    print(f"  CBS CCSD(T) energy: {E_CBS:.12f} Ha")

    return E_CBS


# Calculate CBS energies for all atoms
print("=" * 60)
print("SINGLE ATOM CBS ENERGIES CALCULATION")
print("=" * 60)

atom_energies = {}
for atom_symbol in atoms:
    multiplicity = atoms[atom_symbol]["multiplicity"]
    cbs_energy_ha = calculate_cbs_energy(atom_symbol, multiplicity)
    # Convert to eV (1 Ha = 27.2114 eV)
    cbs_energy_ev = cbs_energy_ha * 27.2114
    atom_energies[atom_symbol] = f"{cbs_energy_ev:.6f}"

# Print results
print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print("Single atom CBS energies in eV:")
print(atom_energies)

# Also print in a more readable format
print("\nDetailed results:")
for atom, energy_ev in atom_energies.items():
    energy_ha = float(energy_ev) / 27.2114
    print(f"{atom:2s}: {energy_ha:.12f} Ha = {energy_ev:>10s} eV")
