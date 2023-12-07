import lightning as L

from zinc.zinc_gta3 import GTA3_ZINC, GTA3_ZINC_Dataset

def main():
    # arguments

    # parameters
    model_params = dict()
    model_params['hidden_dim'] = 64
    model_params['out_dim'] = 64
    model_params['phi'] = 'id'
    model_params['num_heads'] = 4
    model_params['num_layers'] = 2
    model_params['residual'] = True
    model_params['batch_norm'] = True
    model_params['attention_bias'] = True

    train_params = dict()
    train_params['lr'] = 1e-3

    # load data
    train_loader = GTA3_ZINC_Dataset('train')
    model_params['num_types'] = train_loader.get_num_types()

    # load model
    model = GTA3_ZINC(model_params, train_params)

    # train model
    trainer = L.Trainer(max_epochs=3)
    trainer.fit(model=model, train_dataloaders=train_loader)

if __name__ == '__main__':
    main()