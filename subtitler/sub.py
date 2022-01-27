import nemo
import nemo.collections.asr as nemo_asr
import tempfile
import os

#load the model

model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_256_mls")

#download the audio and split it in 5s segments

url = input("url ? : ")

os.system("youtube-dl -f 'bestaudio' {0} -o out".format(url))
os.system("ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 out > duration")

duration = 0.
with open('duration') as f:
    lines = f.readlines()
    duration = float(lines[0])

p = "/home/jovyan/projet-ml/subtitler/audio-"
paths = []

os.system("rm audio-*.wav")

n = int(duration/5)+1

for i in range(n):
    os.system("ffmpeg -i out -acodec pcm_s16le -ac 1 -ar 16000 audio-{0}.wav".format(i))
    paths.append(p+str(i)+'.wav')
    
#Get the transcription
hyp = model.transcribe(paths, return_hypotheses=True)[0][0]

#Split it into words with timesteps
timed_words = []

for token, t in zip(hyp.y_sequence, hyp.timestep):
    syl = model.decoding.decode_ids_to_tokens([int(token)])[0]
    if syl[0]=='▁':
        timed_words.append((syl[1:], t, t))
    else:
        w,b,_ = timed_words[-1]
        timed_words[-1] = (w+syl, b, t)


print(timed_words)
