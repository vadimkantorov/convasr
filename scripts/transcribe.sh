#CHECKPOINT=data/speechcore/demo/JasperNetBig_NovoGrad_lr5e-5_wd1e-3_bs32____finetune_kfold_long_train_long_ftune_1604_0_checkpoint_epoch326_iter0534734.pt

CHECKPOINT=data/speechcore/demo/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs1024____long-train_bs1024_step6_430k_checkpoint_epoch247_iter0446576.pt

#CHECKPOINT=data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt

python3 transcribe.py $@ \
  --checkpoint $CHECKPOINT \
  -i kontur.wav \
  --mono --html \
  --max-segment-duration 4.0

#-i sample10/2017-02-15-osoboe-1908.mp3 sample10/2019-06-21-osoboe-1106.mp3 sample10/2013-08-30-osoboe-1707.mp3 \

#python3 transcribe.py $@ \
#  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt \
#  -i datasets/1C3wrVr7f6k.mp3 \
#  --mono --html
