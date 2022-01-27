import tempfile
import argparse
import subprocess
import math
from pathlib import Path

import nemo
import nemo.collections.asr as nemo_asr
import youtube_dl

TIME_RATIO = 0.08
BLANK_TIME = 2
MIN_BLOCK = 1.5
MAX_BLOCK = 4
SPLIT_CHARACTERS = 47


def load_model(args):
    if args.model:
        return nemo_asr.models.EncDecRNNTBPEModel.restore_from(args.model)
    return nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_en_contextnet_256_mls")

def get_args():
    parser = argparse.ArgumentParser(description='Generate subtitles')
    parser.add_argument('url', help="Url of the video OR a path to the file")
    parser.add_argument('-m', '--model', help='Load a model (.nemo)')

    return parser.parse_args()

def main():
    args = get_args()
    model = load_model(args)

    if Path(args.url).exists():
        video_filename = args.url
    else:
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'outtmpl': '%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mkv',
            }],
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(args.url, download=True)
            video_filename = f"{info['id']}.mkv"
    
    escaped_video_filename = video_filename.replace('"', '\\"')
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("Extracting audio")
        filename = Path(tmpdirname) / 'audio.mp3'
        cmd_extract = f'ffmpeg -i "{escaped_video_filename}" {filename}'
        cmd_proc = subprocess.Popen(cmd_extract, shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert cmd_proc.wait() == 0

        cmd_get_duration = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{filename}"'
        cmd_proc = subprocess.Popen(cmd_get_duration, shell=True, stdout=subprocess.PIPE)
        duration = float(cmd_proc.stdout.read().decode('utf-8').strip())

        paths = []
        n = math.ceil(duration/5)

        for i in range(n):
            print(f'\rExtracting audio segment {i+1}/{n}', end='')
            wav_path = str(Path(tmpdirname) / f'audio-{i}.wav')
            cmd_extract = f'ffmpeg -i "{filename}" -acodec pcm_s16le -ac 1 -ar 16000 {wav_path}'
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

    srt_path = str(Path(video_filename).with_suffix('.srt'))
    write_subtitle_file(srt_path, timed_words)


def to_timestamp(seconds):
    millis = int(seconds*1000)%1000
    sec = int(seconds)%60
    minutes = int(seconds/60)%60
    hours = int(seconds/3600)%100
    return f'{hours:02d}:{minutes:02d}:{sec:02d},{millis:03d}'

def split_blocks_on(blocks, func):
    new_blocks = []
    for b in blocks:
        new_blocks.extend(func(b))
    return new_blocks

def split_blanks(segments):
    blocks = []
    for seg in segments:
        if not blocks or blocks[-1][-1][2] + BLANK_TIME <= seg[1]:
            blocks.append([])
        blocks[-1].append(seg)
    return blocks

def split_duration(segments):
    blocks = []
    for seg in segments:
        if not blocks or seg[2] - blocks[-1][0][1] > MAX_BLOCK:
            blocks.append([])
        blocks[-1].append(seg)
    return blocks

def split_lines(sentence):
    lines = []
    for s in sentence:
        if not lines or len(lines[-1]) + len(s) + 1 > SPLIT_CHARACTERS:
            lines.append(s)
        else:
            lines[-1] = lines[-1] + ' ' + s
    return lines

def write_subtitle_file(srt_path, words):
    blocks = [[(w, t1*TIME_RATIO, t2*TIME_RATIO) for w, t1, t2 in words]]
    blocks = split_blocks_on(blocks, split_blanks)
    blocks = split_blocks_on(blocks, split_duration)

    sentences = []
    for b in blocks:
        words = [seg[0] for seg in b]
        start, end = b[0][1], b[-1][2]
        end = max(end, start + MIN_BLOCK)
        sentences.append([split_lines(words), start, end])
    
    for s, next_s in zip(sentences, sentences[1:]):
        s[2] = min(s[2], next_s[1])
    
    with open(srt_path, 'w') as srt_file:
        for i_part, (lines, start, end) in enumerate(sentences):
            srt_file.write(f'{i_part+1}\n')
            srt_file.write(f'{to_timestamp(start)} --> {to_timestamp(end)}\n')
            for l in lines:
                srt_file.write(f'{l}\n')
            srt_file.write('\n')


if __name__ == "__main__":
    main()