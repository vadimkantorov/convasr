#python3 transcribe.py \
#  --output-path data/sample_ok.transcribe \
#  --checkpoint data/experiments/JasperNet_NovoGrad_lr1e-2_wd1e-3_bs80___jasperbig/checkpoint_epoch02_iter0062500.pt \
#  --data-path ../sample_ok/

python3 transcribe.py \
  --output-path data/kontur_fullrecs.transcribe \
  --checkpoint data/experiments/JasperNet_NovoGrad_lr1e-2_wd1e-3_bs80___jasperbig/checkpoint_epoch02_iter0062500.pt \
  --data-path data/kontur_fullrecs
