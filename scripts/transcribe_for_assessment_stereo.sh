python3 transcribe.py $@ \
	-i /home/yuborovskikh/work/convasr/data/to_transcribe/2_channel_calls -o /home/yuborovskikh/work/convasr/data/to_transcribe/2_channel_calls_out \
         --speakers 0 1 --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs256____youtube_3000h_finetune_2/checkpoint_epoch448_iter0435000.pt --txt

