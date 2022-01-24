import csv
import sentencepiece as spm
import os

csv_path = '/home/jovyan/projet-ml/data/moz-fr/train.tsv'
corpus_path = 'corpus.txt'

with open(csv_path, newline='') as csvfile:
    data = csv.reader(csvfile, delimiter='\t')
    labels = next(data)
    with open(corpus_path, 'w') as corpus:
        for row in data:
            corpus.write(row[2])
            corpus.write('\n')

spm.SentencePieceTrainer.train(input='corpus.txt', model_prefix='tokenizer', vocab_size=1024, character_coverage=1.0, model_type='bpe')
os.system('mv tokenizer.vocab vocab.txt')