# Graph Transformer with Adjacency Aware Attention (GTA3)

## Models

### ZINC model

Use `zinc_main.py` together with a config file. Configs can be found under `config/zinc/`. Reference: [dgl ZINCDataset](https://docs.dgl.ai/generated/dgl.data.ZINCDataset.html#dgl.data.ZINCDataset)

```
zinc_main.py config [--force_reload] [--no_wandb]

positional arguments:
  config              Path to the config file to be used.

optional arguments:
  --force_reload      Will force the dataloader to reload the raw data and preprocess it instead of using cached data.
  --no_wandb          Will not use the WandB logger (useful for debugging).
```

### CLUSTER model

Use `cluster_main.py` together with a config file. Configs can be found under `config/cluster/`. Reference: [dgl CLUSTERDataset](https://docs.dgl.ai/generated/dgl.data.CLUSTERDataset.html)

```
cluster_main.py config [--force_reload] [--no_wandb]

positional arguments:
  config              Path to the config file to be used.

optional arguments:
  --force_reload      Will force the dataloader to reload the raw data and preprocess it instead of using cached data.
  --no_wandb          Will not use the WandB logger (useful for debugging).
```

### Neighborhoodmatch model

Use `nbm_main.py` together with a config file. Configs can be found under `config/nbm/`. Reference: [On the Bottleneck of Graph Neural Networks and its Practical Implications - Uri Alon, Eran Yahav](https://arxiv.org/abs/2006.05205)

```
nbm_main.py config [--force_reload] [--force_regenerate] [--no_wandb]

positional arguments:
  config              Path to the config file to be used.

optional arguments:
  --force_reload      Will force the dataloader to reload the raw data and preprocess it instead of using cached data.
  --force_regenerate  Will force the dataloader to regenerate the raw data.
  --no_wandb          Will not use the WandB logger (useful for debugging).
```


## Setup

### Conda Using Requirements File

TODO: using requirements

### Conda Using Install Commands

The following commands are sufficient to install everything needed to run the models:

```
conda install -c conda-forge lightning
conda install -c dglteam dgl
conda install -c esri einops
conda install -c conda-forge wandb
conda install -c conda-forge scikit-learn
```

Note: The GNN models might require a cuda version of dgl. The available packages can be found under [DGL Installation](https://www.dgl.ai/pages/start.html).

```
# dgl for cuda v12.1 using conda:
conda install -c dglteam/label/cu121 dgl

# the conda installation might not work properly, instead use the default dgl from above and install the cuda part using pip:
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html
```


## GTA3 Model

Base implementation under `gta3/`. Dataset specific implementations under `zinc/`, `cluster/` and `neighborsmatch/`.

## Reproducability
Use the flag --no_wandb to prevent WeightsAndBiasas from attempting to log data.

For the NEIGHBORSMATCH dataset use the `nbm_r_benchmark` branch along with the configs in `config/nbm`

For the CLUSTER and ZINC datasets use the `main` branch for the GTA3 model with learnable alpha and the `local-global` branch for the model with local and global heads. Use the configs `config/<dataset>/gta3_500k[_PE].json` for the corresponding datasets.
