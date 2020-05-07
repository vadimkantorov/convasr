python3 transcribe.py $@ \
	-i data/to_transcribe/uks_wavs -o data/to_transcribe/uks_wavs_out \
         --speakers Consultant Client --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr5e-5_wd1e-3_bs32____finetune_kfold_long_train_long_ftune_1604_0_checkpoint_epoch326_iter0534734.pt --txt --skip-processed --skip-duration 3 


