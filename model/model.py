import yaml

import torch.nn as nn
import nemo
import nemo.collections.asr as nemo_asr

CONFIG_PATH = 'model/config.yaml'

def load_config(path=CONFIG_PATH):
    return yaml.safe_load(open(path))

def load_model(params):
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=params['name'])
    model.change_vocabulary(params['model']['tokenizer']['dir'], params['model']['tokenizer']['type'])

    model.setup_training_data(train_data_config=params['model']['train_ds'])
    model.setup_validation_data(val_data_config=params['model']['validation_ds'])
    model.setup_optimization(optim_config=params['model']['optim'])

    return model

# Freeze the encoder: from https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb
def unfreeze_squeeze_excitation(m):
    if "SqueezeExcite" in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

def unfreeze_batch_norm(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

def freeze_model(model, params):
    if params['freeze']['encoder']:
        model.encoder.freeze()

        if params['freeze']['unfreeze']['squeeze_excitation']:
            model.encoder.apply(unfreeze_squeeze_excitation)
        if params['freeze']['unfreeze']['batch_norm']:
            model.encoder.apply(unfreeze_batch_norm)