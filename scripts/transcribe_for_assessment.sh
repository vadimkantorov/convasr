python3 transcribe.py $@ \
	-i /home/yuborovskikh/work/projs/dataset_tools/chunk_16/chunk_16 -o /home/yuborovskikh/work/projs/dataset_tools/chunk_16/chunk_16/transcripts_old \
         --mono --skip-json --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs256____youtube_3000h_finetune_2/checkpoint_epoch448_iter0435000.pt --txt

#prev best data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs256____youtube_3000h_finetune_2/checkpoint_epoch448_iter0435000.pt
#curr best best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____longtrain_finetune_after_self_train_checkpoint_epoch400_iter0740000.pt
