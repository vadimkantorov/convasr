python3 transcribe.py $@ \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt \
  -i sample10/2012-09-07-osoboe-1705.mp3 \
  --mono --align --html \
  --min-cer 0.0 --max-cer 0.4 --min-duration 0.1 --max-duration 2.0 --align-boundary-words

#python3 transcribe.py $@ \
#  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt \
#  -i datasets/1C3wrVr7f6k.mp3 \
#  --mono --html
