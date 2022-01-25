import csv
import json
import random
from pydub import AudioSegment

AUDIO_PATH = 'data/cvoice/clips/'
TRAIN_PATH = 'data/cvoice/train.tsv'
TEST_PATH = 'data/cvoice/test.tsv'

N_MAX_TRAIN = 20000
N_MAX_TEST = 5000

def gen_dataset(tsv_path, split=None, n_limit=10**20):
    print("Reading config from", tsv_path)
    audios = []
    with open(tsv_path, newline='') as csvfile:
        data = csv.reader(csvfile, delimiter='\t')
        labels = next(data) # ['client_id', 'path', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent']
        for row in data:
            audios.append((AUDIO_PATH + row[1] + '.mp3', row[2]))
    
    random.shuffle(audios)
    print(f"Taking <={n_limit} of {len(audios)} files")
    audios = audios[:n_limit]
    print("Computing audio files size")
    print('', end='')
    audios = [ print(f'\rFile {i+1}/{len(audios)}', end='', flush=True) or {
            "audio_filepath": path,
            "text": text,
            "duration": AudioSegment.from_file(path).duration_seconds,
        } for i, (path, text) in enumerate(audios)
    ]
    print("\nDone")
    if split is not None:
        n = int(len(audios) * split)
        return audios[:n], audios[n:]
    return audios

def store_dataset(dataset, path):
    with open(path, 'w') as f:
        for row in dataset:
            f.write(json.dumps(row) + '\n')

train, validation = gen_dataset(TRAIN_PATH, split=0.8, n_limit=N_MAX_TRAIN)
store_dataset(train, 'datasets/train.json')
store_dataset(validation, 'datasets/validation.json')
store_dataset(gen_dataset(TEST_PATH, n_limit=N_MAX_TEST), 'datasets/test.json')