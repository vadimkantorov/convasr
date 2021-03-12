set -e

python3 train.py $@ \
  --device cuda \
  --fp16 O2 \
  --onnx best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____rerun_finetune_after_self_train_epoch183_iter0290000.pt.12.fp16_masking.onnx \
  --onnx-opset 12 \
  --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____rerun_finetune_after_self_train_epoch183_iter0290000.pt \
