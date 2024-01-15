import torch
import lightning as L
from lightning.pytorch.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import WandbLogger
import json
import os
import os.path as osp
import argparse

from cluster.cluster_gta3 import GTA3_CLUSTER, GTA3_CLUSTER_Dataset
from cluster.cluster_gnn import GNN_CLUSTER, GNN_CLUSTER_DataLoader

from util.lightning_util import StopOnLrCallback

def main():
    # arguments
    parser = argparse.ArgumentParser(description='Main program to train and evaluate models based on the CLUSTER dataset.')
    parser.add_argument('config', type=str,
		                help="Path to the config file to be used.")
    parser.add_argument('--force_reload', action="store_true",
		                help="Will force the dataloader to reload the raw data and preprocess it instead of using cached data.")
    parser.add_argument('--disable_caching', action="store_false",
		                help="Will make the dataloader not to cache the preprocessed data.")
    parser.add_argument('--no_wandb', action="store_true",
		                help="Will not use the WandB logger (useful for debugging).")
    parser.add_argument('--seed', type=int, default=None, help="Manually set the seed for the run.")
    args = parser.parse_args()

    # load the config
    if not osp.exists(args.config):
        print(f"Invalid config file! The file {args.config} does not exist.")
        exit()
    config = None
    with open(args.config) as config_file:
        config = json.load(config_file)
    if config['dataset'] != 'cluster':
        raise ValueError(f"Config is for the wrong dataset! Expecting 'cluster', got {config['dataset']}!")

    # override the seed if we are using one
    if args.seed is not None:
        config['train_params']['seed'] = args.seed

    # set the seed if we are using one
    if config['train_params']['seed'] is not None:
        print(f"Setting manual seed to {config['train_params']['seed']}.")
        L.seed_everything(config['train_params']['seed'])
    torch.set_float32_matmul_precision('medium')

    # load the training data
    if config['model'] == 'gta3':
        pos_enc_dim = config['model_params']['pos_enc_dim'] if 'pos_enc_dim' in config['model_params'] else None
        train_loader = GTA3_CLUSTER_Dataset('train', phi_func=config['model_params']['phi'], pos_enc=config['model_params']['pos_encoding'],
                                            batch_size=config['train_params']['batch_size'], force_reload=args.force_reload,
                                            use_caching=args.disable_caching, pos_enc_dim=pos_enc_dim)
        valid_loader = GTA3_CLUSTER_Dataset('valid', phi_func=config['model_params']['phi'], pos_enc=config['model_params']['pos_encoding'],
                                            batch_size=config['train_params']['batch_size'], force_reload=args.force_reload,
                                            use_caching=args.disable_caching, pos_enc_dim=pos_enc_dim)
        test_loader = GTA3_CLUSTER_Dataset('test', phi_func=config['model_params']['phi'], pos_enc=config['model_params']['pos_encoding'],
                                            batch_size=config['train_params']['batch_size'], force_reload=args.force_reload,
                                            use_caching=args.disable_caching, pos_enc_dim=pos_enc_dim)
    elif config['model'] in ('gcn', 'gat'):
        train_loader = GNN_CLUSTER_DataLoader('train', batch_size=config['train_params']['batch_size'])
        valid_loader = GNN_CLUSTER_DataLoader('valid', batch_size=config['train_params']['batch_size'])
        test_loader = GNN_CLUSTER_DataLoader('test', batch_size=config['train_params']['batch_size'])
    else:
        raise ValueError(f"Unkown model {config['model']} in config file {args.config}!")
    
    config['model_params']['num_out_types'] = train_loader.get_num_out_types()
    config['model_params']['num_in_types'] = train_loader.get_num_in_types()
    config['model_params']['max_num_nodes'] = 190

    # load the model
    if config['model'] == 'gta3':
        model = GTA3_CLUSTER(config['model_params'], config['train_params'])
    elif config['model'] in ('gcn', 'gat'):
        model = GNN_CLUSTER(config['model'], config['model_params'], config['train_params'])
    else:
        raise ValueError(f"Unkown model {config['model']} in config file {args.config}!")

    # train the model
    if not args.no_wandb:
        os.makedirs(config['logging']['save_dir'], exist_ok=True)
        logger = WandbLogger(entity='thrupf', project='gta3-thomas', name=config['logging']['name'], save_dir=config['logging']['save_dir'], log_model=True,)
        logger.log_hyperparams(config)
    else:
        logger = None
    trainer = L.Trainer(
        # max_epochs=config['train_params']['max_epochs'],
        max_time='00:12:00:00',
        logger=logger,
        check_val_every_n_epoch=1,  # needed for lr scheduler
        callbacks=[StopOnLrCallback(lr_threshold=config['train_params']['lr_threshold'], on_val=True)],
        plugins=[SLURMEnvironment(auto_requeue=False)],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # evaluate the model
    trainer.test(model=model, dataloaders=test_loader, verbose=True)

if __name__ == '__main__':
    main()