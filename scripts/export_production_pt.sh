set -e

python3 train.py $@ \
  --onnx best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs256____youtube_3000h_finetune_2_checkpoint_epoch448_iter0435000.onnx \
  --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs256____youtube_3000h_finetune_2_checkpoint_epoch448_iter0435000.pt

