set -e

python3 train.py $@ \
  --checkpoint data/checkpoint_epoch02_iter0065000.pt 
