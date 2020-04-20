set -e

python3 train.py $@ \
  --onnx data/model_16042020_2.onnx \
  --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr5e-5_wd1e-3_bs32____finetune_kfold_long_train_long_ftune_1604_0_checkpoint_epoch326_iter0534734.pt
#  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt 
