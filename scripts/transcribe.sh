python3 transcribe.py $@ \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt \
  -i sample10 \
  --mono --align \
  --min-cer 0.1 --max-cer 0.4 --min-duration 0.1 --max-duration 2.0 --align-boundary-words

