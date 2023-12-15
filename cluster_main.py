import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import json
import os.path as osp
import argparse

from cluster.cluster_gta3 import GTA3_CLUSTER, GTA3_CLUSTER_Dataset

def main():
    # arguments
    parser = argparse.ArgumentParser(description='Main program to train and evaluate models based on the CLUSTER dataset.')
    parser.add_argument('config', type=str,
		                help="Path to the config file to be used.")
    parser.add_argument('--force_reload', action="store_true",
		                help="Will force the dataloader to reload the raw data and preprocess it instead of using cached data.")
    args = parser.parse_args()

    # load the config
    if not osp.exists(args.config):
        print(f"Invalid config file! The file {args.config} does not exist.")
        exit()
    config = None
    with open(args.config) as config_file:
        config = json.load(config_file)

    # set the seed if we are using one
    if config['train_params']['seed'] is not None:
        print(f"Setting manual seed to {config['train_params']['seed']}.")
        torch.manual_seed(config['train_params']['seed'])
    torch.set_float32_matmul_precision('medium')

    # load the training data
    train_loader = GTA3_CLUSTER_Dataset('train', phi_func=config['model_params']['phi'], force_reload=args.force_reload)
    valid_loader = GTA3_CLUSTER_Dataset('valid', phi_func=config['model_params']['phi'], force_reload=args.force_reload)
    config['model_params']['num_classes'] = train_loader.get_num_classes()
    config['model_params']['num_types'] = config['model_params']['num_classes'] + 1

    # load the model
    model = GTA3_CLUSTER(config['model_params'], config['train_params'])

    # train the model
    logger = TensorBoardLogger(save_dir=config['logging']['save_dir'], name=config['logging']['name'])
    trainer = L.Trainer(max_epochs=config['train_params']['max_epochs'], logger=logger, check_val_every_n_epoch=config['train_params']['valid_interval'])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()