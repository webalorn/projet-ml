import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
from ruamel_yaml import YAML
from omegaconf import DictConfig


config_path = 'model/config.yaml'
yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)

params['model']['train_ds']['manifest_filepath'] = train_manifest
params['model']['validation_ds']['manifest_filepath'] = test_manifest

model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=config['name'])
model.change_vocabulary(config['tokenizer']['dir'], config['tokenizer']['type'])

model.setup_optimization(optim_config=params['model']['optim'])
model.setup_training_data(train_data_config=params['model']['train_ds'])
model.setup_validation_data(val_data_config=params['model']['validation_ds'])

trainer = pl.Trainer(gpus=1, max_epochs=1)
trainer.fit(model)

# p = '/home/jovyan/projet-ml/data/libri-dataset/dev-clean/1272/128104/1272-128104-0000.flac'
# txt = model.transcribe([p])
# print(txt)