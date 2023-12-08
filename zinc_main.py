import lightning as L
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

    # load the training data
    train_loader = GTA3_ZINC_Dataset('train', phi_func=config['model_params']['phi'])
    config['model_params']['num_types'] = train_loader.get_num_types()

    # load the model
    model = GTA3_ZINC(config['model_params'], config['train_params'])

    # train the model
    trainer = L.Trainer(max_epochs=config['train_params']['max_epochs'])
    trainer.fit(model=model, train_dataloaders=train_loader)

    # validate the model
    valid_loader = GTA3_ZINC_Dataset('valid', phi_func=config['model_params']['phi'])
    trainer.validate(model=model, dataloaders=valid_loader)


if __name__ == '__main__':
    main()