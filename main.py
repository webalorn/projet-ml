
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
    freeze_model(model, params)
    
    print("\n========== Start FIT")
    trainer.fit(model)

    print("\n========== Done fitting")
    Path('checkpoints').mkdir(exist_ok=True)
    model.save_to(f"checkpoints/{params['name']}_fr_{params['freeze']['encoder']}_{params['freeze']['unfreeze']['squeeze_excitation']}_{params['freeze']['unfreeze']['batch_norm']}_{start_time}.nemo")

def test(args):
    assert args.model, "You should load a model (-m option)"
    params = load_config()
    model = load_model(params, args).eval()

    if args.audio:
        predictions = model.transcribe(args.audio)
        for pred, audio in zip(predictions[0], args.audio):
            print(f"# Fichier {audio}")
            print(pred)
    else:
        ds = get_dataloader(model, args.dataset_test)
        ds_size = len(ds)
        print()
        wer_error = nemo.collections.asr.metrics.rnnt_wer_bpe.RNNTBPEWER(
            decoding=model.decoding, log_prediction=False)

        for i_batch, test_batch in enumerate(ds):
            print(f'\rBatch {i_batch+1}/{ds_size}', end=('\n' if args.verbose else ''))
            targets = test_batch[2]
            targets_lengths = test_batch[3]
            encoded, encoded_len = model.forward(
                input_signal=test_batch[0].cuda(), input_signal_length=test_batch[1].cuda()
            )
            best_hyp, _ = model.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len)

            if args.verbose:
                for label_code, n, predicted in zip(targets, targets_lengths, best_hyp):
                    label = model.decoding.decode_tokens_to_str(label_code.numpy()[:n])
                    print("Label:", label)
                    print('-> Predicted:', predicted)

            wer_error(encoded, encoded_len, targets, targets_lengths)
            del encoded, test_batch, best_hyp

        wer = wer_error.compute()[0].item()
        print(f"\n\nWER on test dataset: {wer*100:.3f}%")


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command', choices=['train', 'test'])
    parser.add_argument('-c', '--checkpoint', help='Load a checkpoint')
    parser.add_argument('-m', '--model', help='Load a model (.nemo)')
    parser.add_argument('-a', '--audio', nargs='+', default=[], help='Audio files that will be transcribed (test command)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-dt', '--dataset_test', choices=['train', 'val', 'test'], help='Dataset used with the set command', default='test')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'test':
        test(args)