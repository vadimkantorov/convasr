# convasr
Baseline convolutional ASR system in PyTorch

# Dependencies
PyTorch, NumPy, SciPy, NVidia Apex (for fp16 training)
```shell
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
```

# data file format
CSV (comma-separated) with 3 columns without header:
1. Full path to the audio wav file (mono, 16 Khz)
2. Transcript
3. Duration in seconds
