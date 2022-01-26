
from datetime import datetime
from pathlib import Path
import argparse

import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager

from model.model import *

# CUR_PATH = str(Path('.').absolute())
# class NewAppState(nemo.utils.AppState):
#     @property
#     def nemo_file_folder(self) -> str:
#         return CUR_PATH
# nemo.utils.AppState = NewAppState

# f0 = nemo.core.connectors.save_restore_connector.SaveRestoreConnector.register_artifact
# def f1(s, model, config_path: str, src: str, verify_src_exists: bool = True):
#     print("================================================= F1")
#     print(config_path, src, verify_src_exists)
#     f0(s, model, config_path, src, verify_src_exists)
# nemo.core.connectors.save_restore_connector.SaveRestoreConnector.register_artifact = f1

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
    params = load_config()
    # model = load_model(params, args)
    # trainer = pl.Trainer(**params['trainer'])
    # exp_manager(trainer, params.get("exp_manager", None))

    # model.restore_from("nemo_experiments/stt_en_contextnet_256_mls/True_True_True/checkpoints/stt_en_contextnet_256_mls.nemo")
    # model = nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel.load_from_checkpoint(checkpoint_path="nemo_experiments/stt_en_contextnet_256_mls/checkpoints/stt_en_contextnet_256_mls--val_wer=0.6610-epoch=19-last.ckpt")

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
        break


def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('command', choices=['train', 'test'])
    parser.add_argument('-c', '--checkpoint', help='Load a checkpoint')

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