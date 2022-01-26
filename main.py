
from datetime import datetime
from pathlib import Path
import argparse

import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager

from model.model import *


def train(args):
    start_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    params = load_config()
    model = load_model(params)
    trainer = pl.Trainer(**params['trainer'])
    exp_manager(trainer, params.get("exp_manager", None))
    
    print("\n========== Start FIT")
    trainer.fit(model)

    print("\n========== Done fitting")
    Path('checkpoints').mkdir(exist_ok=True)
    model.save_to(f"checkpoints/{params['name']}_fr_{params['freeze']['encoder']}_{params['freeze']['unfreeze']['squeeze_excitation']}_{params['freeze']['unfreeze']['batch_norm']}_{start_time}.nemo")

def test(args):
    pass

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command', choices=['train', 'test'])

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)

# p = '/home/jovyan/projet-ml/data/libri-dataset/dev-clean/1272/128104/1272-128104-0000.flac'
# txt = model.transcribe([p])
# print(txt)