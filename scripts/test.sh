CUDA_VISIBLE_DEVICES=1 python3 train.py --lang ru \
  --model Wav2LetterRu \
  --val-batch-size 20 --val-data-path sample_ok/sample_ok.convasr.csv \
  --checkpoint ck3.pt
