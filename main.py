import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
import yaml
from omegaconf import DictConfig


CONFIG_PATH = 'model/config.yaml'

with open(CONFIG_PATH) as f:
    params = yaml.safe_load(f)

model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=params['name'])
model.change_vocabulary(params['model']['tokenizer']['dir'], params['model']['tokenizer']['type'])

model.setup_training_data(train_data_config=params['model']['train_ds'])
model.setup_validation_data(val_data_config=params['model']['validation_ds'])
model.setup_optimization(optim_config=params['model']['optim'])

trainer = pl.Trainer(**params['trainer'])

print("\n========== Start FIT")
trainer.fit(model)

# p = '/home/jovyan/projet-ml/data/libri-dataset/dev-clean/1272/128104/1272-128104-0000.flac'
# txt = model.transcribe([p])
# print(txt)