import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger
import json
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
    parser.add_argument('--no_wandb', action="store_true",
		                help="Will not use the WandB logger (useful for debugging).")
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

    # set the seed if we are using one
    if config['train_params']['seed'] is not None:
        print(f"Setting manual seed to {config['train_params']['seed']}.")
        torch.manual_seed(config['train_params']['seed'])
    torch.set_float32_matmul_precision('medium')

    # load the training data
    if config['model'] == 'gta3':
        pos_enc_dim = config['model_params']['pos_enc_dim'] if 'pos_enc_dim' in config['model_params'] else None
        train_loader = GTA3_CLUSTER_Dataset('train', phi_func=config['model_params']['phi'], pos_enc=config['model_params']['pos_encoding'],
                                            batch_size=config['train_params']['batch_size'], force_reload=args.force_reload, pos_enc_dim=pos_enc_dim)
        valid_loader = GTA3_CLUSTER_Dataset('valid', phi_func=config['model_params']['phi'], pos_enc=config['model_params']['pos_encoding'],
                                            batch_size=config['train_params']['batch_size'], force_reload=args.force_reload, pos_enc_dim=pos_enc_dim)
        test_loader = GTA3_CLUSTER_Dataset('test', phi_func=config['model_params']['phi'], pos_enc=config['model_params']['pos_encoding'],
                                            batch_size=config['train_params']['batch_size'], force_reload=args.force_reload, pos_enc_dim=pos_enc_dim)
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
        logger = WandbLogger(entity='gta3', project='gta3', name=config['logging']['name'], save_dir=config['logging']['save_dir'], log_model='all',)
        logger.log_hyperparams(config)
    else:
        logger = None
    trainer = L.Trainer(
        max_epochs=config['train_params']['max_epochs'],
        logger=logger,
        check_val_every_n_epoch=config['train_params']['valid_interval'],
        callbacks=[StopOnLrCallback(lr_threshold=1e-6, on_val=True)],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    # evaluate the model
    trainer.test(model=model, test_dataloaders=test_loader, verbose=True)

if __name__ == '__main__':
    main()