set -e

python3 train.py $@ \
  --onnx best_checkpoints/JasperNetBig_NovoGrad_lr2e-5_wd1e-3_bs512_slow_fine_tune_after_uncertainty_self_train_checkpoint_epoch268_iter0370000.12.fp16.onnx --onnx-opset 12 \
  --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr2e-5_wd1e-3_bs512_slow_fine_tune_after_uncertainty_self_train_checkpoint_epoch268_iter0370000.pt \
  --onnx-dot-file best_checkpoints/model.dot \
  --fp16 O2
