python3 transcribe.py $@ \
	-i domain_set/transcripts/16082020/valset_16082020.csv.json -o data/valset_16082020_logits/ \
         --logits --mono --checkpoint best_checkpoints/JasperNetBig_NovoGrad_lr1e-4_wd1e-3_bs512____rerun_finetune_after_self_train_epoch183_iter0290000.pt --output-txt --output-csv
