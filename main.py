
from datetime import datetime
from pathlib import Path
import argparse

import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager

from model.model import *

def train(args):
    start_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
    params = load_config()
    model = load_model(params, args)
    trainer = pl.Trainer(**params['trainer'])
    exp_manager(trainer, params.get("exp_manager", None))
    
    print("\n========== Start FIT")
    trainer.fit(model)

    print("\n========== Done fitting")
    Path('checkpoints').mkdir(exist_ok=True)
    model.save_to(f"checkpoints/{params['name']}_fr_{params['freeze']['encoder']}_{params['freeze']['unfreeze']['squeeze_excitation']}_{params['freeze']['unfreeze']['batch_norm']}_{start_time}.nemo")

def test(args):
    assert args.model, "You should load a model (-m option)"
    params = load_config()
    model = load_model(params, args)
    model.eval()

    with torch.no_grad():
        if args.audio:
            predictions = model.transcribe(args.audio)
            for pred, audio in zip(predictions[0], args.audio):
                print(f"# Fichier {audio}")
                print(pred)
        else:
            for test_batch in model.test_dataloader():
                targets = test_batch[2]
                targets_lengths = test_batch[3]
                encoded, encoded_len = model.forward(
                    input_signal=test_batch[0].cuda(), input_signal_length=test_batch[1].cuda()
                )
                best_hyp, _ = model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len)

                for label_code, n, predicted in zip(targets, targets_lengths, best_hyp):
                    label = model.decoding.decode_tokens_to_str(label_code.numpy()[:n])
                    print("Label:", label)
                    print('-> Predicted:', predicted)

                del encoded, test_batch, best_hyp


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command', choices=['train', 'test'])
    parser.add_argument('-c', '--checkpoint', help='Load a checkpoint')
    parser.add_argument('-m', '--model', help='Load a model (.nemo)')
    parser.add_argument('-a', '--audio', nargs='+', default=[], help='Audio files that will be transcribed (test command)')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)