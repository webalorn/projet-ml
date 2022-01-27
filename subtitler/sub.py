import tempfile
import argparse
import subprocess
import math
from pathlib import Path

import nemo
import nemo.collections.asr as nemo_asr
import youtube_dl


def load_model():
    return nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_256_mls")

def get_args():
    parser = argparse.ArgumentParser(description='Generate subtitles')
    parser.add_argument('url', help="Url of the audio/video")

    return parser.parse_args()

def main():
    args = get_args()
    model = load_model()

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(args.url, download=True)
        filename = f"{info['id']}.mp3"

    escaped_filename = filename.replace('"', '\\"')
    cmd_get_duration = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{escaped_filename}"'
    cmd_proc = subprocess.Popen(cmd_get_duration, shell=True, stdout=subprocess.PIPE)
    duration = float(cmd_proc.stdout.read().decode('utf-8').strip())

    with tempfile.TemporaryDirectory() as tmpdirname:
        paths = []
        n = math.ceil(duration/5)

        for i in range(n):
            print(f'\rExtracting audio segment {i+1}/{n}', end='')
            wav_path = str(Path(tmpdirname) / f'audio-{i}.wav')
            cmd_extract = f'ffmpeg -i "{escaped_filename}" -acodec pcm_s16le -ac 1 -ar 16000 {wav_path}'
            cmd_proc = subprocess.Popen(cmd_extract, shell=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            assert cmd_proc.wait() == 0
            paths.append(wav_path)
        print()

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

def write_subtitle_file(filename, data):
    filename = str(Path(filename).with_suffix('.srt'))

if __name__ == "__main__":
    main()
    # write_subtitle_file('sub.mp3', [('i', 1, 1), ('would', 3, 3), ('like', 5, 5), ('to', 8, 8), ('check', 9, 11), ('for', 13, 13), ('traps', 15, 18), ('or', 20, 20), ('right', 22, 22), ('go', 25, 25), ('henroll', 27, 33), ('fourteen', 35, 40), ('as', 43, 43), ('a', 45, 45), ('check', 47, 49), ('for', 50, 50), ('traps', 52, 55), ('to', 55, 55), ('hit', 57, 58), ('the', 59, 59), ('trip', 60, 61), ('wire', 62, 64), ('which', 66, 66), ('opens', 68, 70), ('up', 72, 72), ('a', 73, 73), ('death', 74, 74), ('spike', 77, 79), ('trap', 81, 83), ('right', 84, 84), ('below', 87, 89), ('your', 90, 90), ('feet', 92, 92), ('and', 95, 95), ('your', 97, 97), ('characters', 100, 104), ('killed', 106, 109), ('sh', 112, 112), ('he', 118, 118), ("doesn't", 120, 122), ('know', 124, 124), ('i', 126, 126), ('know', 128, 128), ('this', 130, 130), ('trick', 133, 135)])