python3 transcribe.py $@ \
  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt \
  -i data/speechcore/bugs/bug_5168 \
  --mono --align --html \
  --max-segment-duration 4.0

#-i sample10/2017-02-15-osoboe-1908.mp3 sample10/2019-06-21-osoboe-1106.mp3 sample10/2013-08-30-osoboe-1707.mp3 \

#python3 transcribe.py $@ \
#  --checkpoint data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt \
#  -i datasets/1C3wrVr7f6k.mp3 \
#  --mono --html
