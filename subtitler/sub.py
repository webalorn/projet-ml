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

p = "/home/jovyan/projet-ml/subtitler/audio.wav"
paths = [p]

os.system("rm audio*.wav")

n = int(duration/5)+1
os.system("ffmpeg -i out -acodec pcm_s16le -ac 1 -ar 16000 audio0.wav")
os.system("ffmpeg -f concat -safe 0 -i concat audio.wav")
 
#Get the transcription
hyp = model.transcribe(paths, return_hypotheses=True)[0][0]

#Split it into words with timesteps

lines = ['00:00:00,000 --> 00:00:05,000\n', '']

tmax = hyp.timestep[-1]

timed_words = []
for token, t in zip(hyp.y_sequence, hyp.timestep):
    syl = model.decoding.decode_ids_to_tokens([int(token)])[0]
    if syl[0]=='▁':
        timed_words.append((syl[1:], t, t))
    else:
        w,b,_ = timed_words[-1]
        timed_words[-1] = (w+syl, b, t)

extract = 0
for (w, b, e) in timed_words:
    idx_extract = (n*e)//tmax
    if idx_extract > extract:
        extract += 1
        i = extract
        lines.append('\n\n00:{0}:{2},000 --> 00:{1}:{3},000\n'.format(str(i//12%60).zfill(2), str((i+1)//12%60).zfill(2),
            str(5*i%60).zfill(2), str(5*(i+1)%60)).zfill(2))
        lines.append('')
    lines[-1] += w + " "

with open('caps.srt', 'w') as f:
    for line in lines:
        f.write(line)