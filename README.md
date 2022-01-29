# projet-ml


## Dependencies

You may want to install these in a virtual environment or Conda environment.
```bash
python3 -m pip install Cython
python3 -m pip install nemo_toolkit[all] youtube_dl
```

If you encounter the following error (or similar) when running any script in this repository:
```
    File "pesq/cypesq.pyx", line 1, in init cypesq
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```
you can try reinstalling pesq without the pip cache:
```bash
python3 -m pip uninstall pesq
python3 -m pip install --no-cache-dir pesq
```

You also need `ffmpeg`.

## Sources

- [NVIDIA NeMo Toolkit](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/index.html)
- [ContextNet](https://arxiv.org/abs/2005.03191)
