python3 transcribe.py $@ \
	-i data/input -o data/output \
        --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr5e-5_wd1e-3_bs32____finetune_kfold_long_train_long_ftune_1603_0_checkpoint_epoch206_iter0248969.pt \
	--html --speakers 0 1 --max-segment-duration 60
	#--checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr5e-4_wd1e-3_bs32____long_train_finetune5e4_checkpoint_epoch12_iter0138025.pt \
        #--align-boundary-words --num-speakers 2 --gap 0.1- --cer 0.1-0.4 --duration 2.0-4.0 --max-segment-duration 4.0 \
	#--checkpoint data/experiments/JasperNetBig_NovoGrad_lr5e-4_wd1e-3_bs32____microval_finetune_2/checkpoint_epoch09_iter0136702.pt
#--data-path data/kontur_fullrecs --output-path data/kontur_fullrecs_ \

#-i sample10/2017-02-15-osoboe-1908.mp3 sample10/2019-06-21-osoboe-1106.mp3 sample10/2013-08-30-osoboe-1707.mp3 \

#python3 transcribe.py $@ \
#  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt \
#  -i datasets/1C3wrVr7f6k.mp3 \
#  --mono --html
