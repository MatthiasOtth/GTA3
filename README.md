# Graph Transformer with Adjacency Aware Attention (GTA3)

Useful links:

- [Overleaf project](https://www.overleaf.com/project/656890c8c129707cf405b59e)


## TODOs

- [ ] Implement neighbors match problem ([github](https://github.com/tech-srl/bottleneck))
- [ ] Define configs resulting in ~500K parameters


## Models

### ZINC model

Default config can be found under `config/zinc/gta3_default.json`. Reference: [dgl ZINCDataset](https://docs.dgl.ai/generated/dgl.data.ZINCDataset.html#dgl.data.ZINCDataset)

```
usage: zinc_main.py [-h] [--force_reload] config

Main program to train and evaluate models based on the ZINC dataset.

positional arguments:
  config          Path to the config file to be used.

optional arguments:
  -h, --help      show this help message and exit
  --force_reload  Will force the dataloader to reload the raw data and preprocess it instead of using cached data.
```

### CLUSTER model

Default config can be found under `config/zinc/gta3_default.json`. Reference: [dgl CLUSTERDataset](https://docs.dgl.ai/generated/dgl.data.CLUSTERDataset.html)

```
usage: cluster_main.py [-h] [--force_reload] config

Main program to train and evaluate models based on the CLUSTER dataset.

positional arguments:
  config          Path to the config file to be used.

optional arguments:
  -h, --help      show this help message and exit
  --force_reload  Will force the dataloader to reload the raw data and preprocess it instead of using cached data.
```


## Cluster Setup (not the CLUSTER dataset)

### Conda Using Requirements File

TODO: using requirements

### Conda Using Install Commands

The following commands are sufficient to install everything needed to run the models:

```
conda install -c conda-forge lightning
conda install -c dglteam dgl
conda install -c esri einops
```


## GTA3 Model

Base implementation under `gta3/`. Dataset specific implementations under `zinc/` and `cluster/`.