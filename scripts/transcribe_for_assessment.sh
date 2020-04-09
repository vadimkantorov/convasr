python3 transcribe.py $@ \
	-i data/to_transcribe/chunk_10 -o data/to_transcribe/chunk_10_out \
        --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr5e-5_wd1e-3_bs32____finetune_kfold_long_train_long_ftune_1603_0_checkpoint_epoch206_iter0248969.pt \
	--txt --mono
