# Comp Chem Tools

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

This package provides common computational chemistry workflows.

## License

This code is provided under the [MIT license](LICENSE).


## Development

This package uses the [pixi](https://pixi.sh/) package manager to manage dependencies.

To setup a development environment, install `pixi` and then run the following in the root folder of the project:

```
pixi shell
```

This will automatically create a new environment in `.pixi/envs/default` that will have all dependencies required
to run and test the project.

### Testing

To test the code, run 

```
pixi run test
```

### Linting & Formating

The project uses [ruff](https://docs.astral.sh/ruff/) to lint and format the code.

To format the code, run

```
pixi run format
```

To check the code using the linters, use

```
pixi run lint
```

To fix any issues automatically, run

```
pixi run fix
```

```
NUM_CPUS=4 python3 cct/workflows/conformers.py --input data/ref_ligand.sdf --output data/out.sdf --mode reckless --qm_method AIMNet2
NUM_CPUS=4 RUN_LOCAL=True python3 benchmarks/tautbase.py
mayr https://pubs.acs.org/doi/abs/10.1021/jo201562f
fukui https://figshare.com/articles/dataset/Atom_Condensed_Fukui_Functions_Calculated_for_2973_Organic_Molecules/1400514?utm_source=chatgpt.com
```