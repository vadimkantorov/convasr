spotty run -c scripts/spotty.yaml train \
 -p ARGS="--verbose --lang ru \
  --model Wav2LetterRu \
  --train-batch-size 40 --val-batch-size 40 \
  --lr 1e-2 --optimizer SGD \
  --train-data-path /root/convasr/data/clean_train.csv \
  --val-data-path /root/convasr/data/clean_val.csv \
  --val-iteration-interval 10 \
  --epochs 1 
"
