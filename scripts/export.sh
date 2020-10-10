set -e

python3 train.py $@ \
  --onnx data/model.onnx --onnx-opset 12 #\
#  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt
