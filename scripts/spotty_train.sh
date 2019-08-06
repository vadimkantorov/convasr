spotty run -c scripts/spotty.yaml train \
 -p CUDA_VISIBLE_DEVICES=0,1,2,3 \
 -p ARGS="--verbose --lang ru \
  --model Wav2LetterRu \
  --train-batch-size 40 --val-batch-size 40 \
  --lr 1e-2 --optimizer SGD \
  --train-data-path /root/convasr/data/clean_train.csv \
  --val-data-path /root/convasr/data/clean_val.csv \
  --val-iteration-interval 100 \
  --epochs 1 
"
