import sys

if len(sys.argv) < 2:
    sys.stderr.write(f"Usage: python3 {sys.argv[0]} <model file> [audio files...]\n")
    sys.exit(1)

import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(sys.argv[1])
if len(sys.argv) >= 3:
    txt = model.transcribe(sys.argv[2:])
    print(f"Transcription of {sys.argv[2]}: {txt}")
else:
    while True:
        fname = input("File to transcribe (empty to finish): ")
        if fname == "":
            break
        txt = model.transcribe([fname])
        print(f"Transcription of {fname}: {txt}")
