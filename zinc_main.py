import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import json
import os.path as osp
import argparse

from zinc.zinc_gta3 import GTA3_ZINC, GTA3_ZINC_Dataset

def main():
    # arguments
    parser = argparse.ArgumentParser(description='Main program to train and evaluate models based on the ZINC dataset.')
    parser.add_argument('config', type=str,
		                help="Path to the config file to be used.")
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
    train_loader = GTA3_ZINC_Dataset('train', phi_func=config['model_params']['phi'])
    valid_loader = GTA3_ZINC_Dataset('valid', phi_func=config['model_params']['phi'])
    config['model_params']['num_types'] = train_loader.get_num_types()

    # load the model
    model = GTA3_ZINC(config['model_params'], config['train_params'])

    # train the model
    logger = TensorBoardLogger(save_dir=config['logging']['save_dir'], name=config['logging']['name'])
    trainer = L.Trainer(max_epochs=config['train_params']['max_epochs'], logger=logger, check_val_every_n_epoch=config['train_params']['valid_interval'])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()