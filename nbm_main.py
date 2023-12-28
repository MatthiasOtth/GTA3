import torch
import lightning as L
from pytorch_lightning.loggers import WandbLogger
import json
import os.path as osp
import argparse

from neighborsmatch.nbm_gta3 import GTA3_NBM, GTA3_NBM_Dataset

def main():
    # arguments
    parser = argparse.ArgumentParser(description='Main program to train and evaluate models based on the neighborhood match problem.')
    parser.add_argument('config', type=str,
		                help="Path to the config file to be used.")
    parser.add_argument('--force_reload', action="store_true",
		                help="Will force the dataloader to reload the raw data and preprocess it instead of using cached data.")
    parser.add_argument('--force_regenerate', action="store_true",
		                help="Will force the dataloader to regenerate the raw data.")
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

    # set the seed if we are using one
    if config['train_params']['seed'] is not None:
        print(f"Setting manual seed to {config['train_params']['seed']}.")
        torch.manual_seed(config['train_params']['seed'])
    torch.set_float32_matmul_precision('medium')

    # load the training data
    train_loader = GTA3_NBM_Dataset('train', phi_func=config['model_params']['phi'], tree_depth=config['train_params']['tree_depth'], 
                                    batch_size=config['train_params']['batch_size'], force_reload=args.force_reload, 
                                    force_regenerate=args.force_regenerate, generator_seed=config['train_params']['seed'])
    valid_loader = GTA3_NBM_Dataset('valid', phi_func=config['model_params']['phi'], tree_depth=config['train_params']['tree_depth'], 
                                    batch_size=config['train_params']['batch_size'], force_reload=args.force_reload)
    config['model_params']['num_in_types'] = train_loader.get_num_in_types()
    config['model_params']['num_out_types'] = train_loader.get_num_out_types()

    # load the model
    model = GTA3_NBM(config['model_params'], config['train_params'])

    # train the model
    if not args.no_wandb:
        logger = WandbLogger(project='gta3', log_model='all', save_dir=config['logging']['save_dir'], name=config['logging']['name'])
        logger.log_hyperparams(config)
    else:
        logger = None
    trainer = L.Trainer(max_epochs=config['train_params']['max_epochs'], logger=logger, check_val_every_n_epoch=config['train_params']['valid_interval'])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()