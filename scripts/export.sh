set -e

python3 train.py $@ \
  --onnx data/model.onnx \
  --checkpoint data/checkpoint_epoch02_iter0065000.pt 
