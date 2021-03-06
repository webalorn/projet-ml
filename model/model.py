import yaml

import torch.nn as nn
import nemo
import nemo.collections.asr as nemo_asr

CONFIG_PATH = 'model/config.yaml'

def load_config(path=CONFIG_PATH):
    return yaml.safe_load(open(path))

def load_model(params, args):
    if args and args.model:
        model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.model)
        params['exp_manager']['resume_if_exists'] = False
    else:
        model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=params['name'])
        model.change_vocabulary(params['model']['tokenizer']['dir'], params['model']['tokenizer']['type'])

    model.setup_training_data(train_data_config=params['model']['train_ds'])
    model.setup_validation_data(val_data_config=params['model']['validation_ds'])
    model.setup_test_data(test_data_config=params['model']['test_ds'])
    model.setup_optimization(optim_config=params['model']['optim'])

    return model.cpu()#.cuda()

def get_dataloader(model, dataset):
    if dataset == 'train':
        return model.train_dataloader()
    elif dataset == 'test':
        return model.test_dataloader()
    elif dataset == 'val':
        return model.val_dataloader()
    assert False

# Freeze the encoder: from https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_CTC_Language_Finetuning.ipynb
def _apply_unfreeze(m):
    print("Unfreezing", m)
    m.train()
    for param in m.parameters():
        param.requires_grad_(True)

def unfreeze_squeeze_excitation(m):
    if "SqueezeExcite" in type(m).__name__:
        _apply_unfreeze(m)

def unfreeze_batch_norm(m):
    if type(m) == nn.BatchNorm1d:
        _apply_unfreeze(m)

def freeze_model(model, params):
    if params['freeze']['encoder']:
        print("===== Freezing encoder")
        model.encoder.freeze()

        if params['freeze']['unfreeze']['squeeze_excitation']:
            model.encoder.apply(unfreeze_squeeze_excitation)
        if params['freeze']['unfreeze']['batch_norm']:
            model.encoder.apply(unfreeze_batch_norm)
        
        unfreeze_blocks = params['freeze']['unfreeze']['encoder_blocks']
        last_block = len(model.encoder.encoder)-1
        for k in range(last_block, last_block-unfreeze_blocks, -1):
            _apply_unfreeze(model.encoder.encoder[k])