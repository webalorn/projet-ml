import nemo
import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_256_mls")

p = '/home/jovyan/projet-ml/data/libri-dataset/dev-clean/1272/128104/1272-128104-0000.flac'
txt = model.transcribe([p])
print(txt)