set -e

python3 train.py $@ \
  --onnx data/best_domain_model_24122019.onnx \
  --checkpoint  best_checkpoints/JasperNetBig_NovoGrad_lr1e-3_wd1e-3_bs8___bpe_bpe_bi200_conv1d_epilogue40_finetune_decays_newval_freeze7_lr001_bs8_checkpoint_epoch09_iter0010000.pt
