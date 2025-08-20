from rdkit.Chem import rdDetermineBonds


def determine_bonds(mol, charge: int = 0):
    rdDetermineBonds.DetermineConnectivity(mol)
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
    except Exception as e:
        print(repr(e))
