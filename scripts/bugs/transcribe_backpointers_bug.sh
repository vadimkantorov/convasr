python3 transcribe.py $@ \
	 -i /home/yuborovskikh/work/convasr/data/youtube/d.json \
	 -o /home/yuborovskikh/work/convasr/data/debug_out \
	 --join-transcript \
	 --align \
	 --pack-backpointers \
	--transcribe-first-n-sec 3600 --batch-time-padding-multiple 1 --max-segment-duration 4.0 \
         --mono --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs1024____long-train_bs1024_step6_430k_checkpoint_epoch247_iter0446576.pt

