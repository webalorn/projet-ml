import nemo
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_256_mls")

# trainer = pl.Trainer(gpus=1, max_epochs=1)
# trainer.fit(model)

model.change_vocabulary('tokens', 'bpe')

# p = '/home/jovyan/projet-ml/data/libri-dataset/dev-clean/1272/128104/1272-128104-0000.flac'
# txt = model.transcribe([p])
# print(txt)