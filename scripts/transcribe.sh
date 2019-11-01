#python3 transcribe.py \
#  --output-path data/sample_ok.transcribe \
#  --checkpoint data/experiments/JasperNet_NovoGrad_lr1e-2_wd1e-3_bs80___jasperbig/checkpoint_epoch02_iter0062500.pt \
#  --data-path ../sample_ok/

python3 transcribe.py \
  --data-path data/2019.10.19_calls \
  --output-path data/2019.10.19_calls_decoded \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs80/checkpoint_epoch02_iter0057500.pt \
  --decoder BeamSearchDecoder --beam-width 5000 --lm chats_05_prune.binary  #charlm/chats_06_noprune_char.binary # #--lm data/ru_wiyalen_no_punkt.arpa.binary 

#python3 transcribe.py \
#  --output-path data/5ZjeFwlRB.wav.transcribe \
#  --data-path data/5ZjeFwlRB.wav \
#  --checkpoint data/experiments/JasperNet_NovoGrad_lr1e-2_wd1e-3_bs80___jasperbig/checkpoint_epoch02_iter0062500.pt 
