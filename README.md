NUM_CPUS=4 python3 cct/workflows/conformers.py --input data/ref_ligand.sdf --output data/out.sdf --mode reckless --qm_method AIMNet2
NUM_CPUS=4 RUN_LOCAL=True python3 benchmarks/tautbase.py



mayr https://pubs.acs.org/doi/abs/10.1021/jo201562f
fukui https://figshare.com/articles/dataset/Atom_Condensed_Fukui_Functions_Calculated_for_2973_Organic_Molecules/1400514?utm_source=chatgpt.com