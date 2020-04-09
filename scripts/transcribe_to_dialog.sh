python3 transcribe.py $@ \
	-i data/input -o data/output \
        --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr5e-5_wd1e-3_bs32____finetune_kfold_long_train_long_ftune_1603_0_checkpoint_epoch206_iter0248969.pt \
	--html --speakers 0 1 --max-segment-duration 60
