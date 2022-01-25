import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager
import torch.nn as nn
import pytorch_lightning as pl
import yaml
from datetime import datetime
from omegaconf import DictConfig

FREEZE_ENCODER = True
UNFREEZE_SQUEEZE_EXCITATION = True
UNFREEZE_BATCH_NORM = True

DATE = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")

CONFIG_PATH = 'model/config.yaml'

with open(CONFIG_PATH) as f:
    params = yaml.safe_load(f)

model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=params['name'])
model.change_vocabulary(params['model']['tokenizer']['dir'], params['model']['tokenizer']['type'])

model.setup_training_data(train_data_config=params['model']['train_ds'])
model.setup_validation_data(val_data_config=params['model']['validation_ds'])
model.setup_optimization(optim_config=params['model']['optim'])

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

if FREEZE_ENCODER:
    model.encoder.freeze()
    if UNFREEZE_SQUEEZE_EXCITATION:
        model.encoder.apply(unfreeze_squeeze_excitation)
    if UNFREEZE_BATCH_NORM:
        model.encoder.apply(unfreeze_batch_norm)

trainer = pl.Trainer(**params['trainer'])

exp_manager(trainer, params.get("exp_manager", None))
print("\n========== Start FIT")
trainer.fit(model)

print("\n========== Done fitting")
model.save_to(f"{params['name']}_fr_{FREEZE_ENCODER}_{UNFREEZE_SQUEEZE_EXCITATION}_{UNFREEZE_BATCH_NORM}_{DATE}.nemo")

# p = '/home/jovyan/projet-ml/data/libri-dataset/dev-clean/1272/128104/1272-128104-0000.flac'
# txt = model.transcribe([p])
# print(txt)