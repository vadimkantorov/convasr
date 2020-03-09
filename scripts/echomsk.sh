set -e 

export CUDA_VISIBLE_DEVICES=0

ECHOMSK=data/personalno_20000101_20191231.txt.json.gz
CHECKPOINT=data/experiments/JasperNetBig_NovoGrad_lr1e-2_wd1e-3_bs256____fp16O2/checkpoint_epoch05_iter0040000.pt

SAMPLE=10
DATASET=sample10
DATASET_TRANSCRIBE=transcribe.$DATASET
DATASET_SUBSET=subset.$DATASET
DATASET_CUT=cut.$DATASET
TRANSCRIBE='--mono --align --max-segment-duration 4.0 --html'
SUBSET='--align-boundary-words --num-speakers 1 --gap 0.1- --cer 0.1-0.4 --duration 2.0-4.0'
CUT='--dilate 0.02 --strip='
SAMPLE_RATE=8000

python3 datasets/echomsk.py "$ECHOMSK" --name $DATASET --sample $SAMPLE
wget --no-clobber -i "$DATASET/$DATASET.txt" -P "$DATASET"

python3 transcribe.py --checkpoint "$CHECKPOINT" -i "$DATASET" -o "$DATASET_TRANSCRIBE" $TRANSCRIBE

python3 tools.py subset -i $DATASET_TRANSCRIBE -o $DATASET_SUBSET.json $SUBSET

python3 tools.py cut -i $DATASET_SUBSET.json -o $DATASET_CUT $CUT

python3 vis.py audiosample -i $DATASET_CUT/$DATASET_CUT.json -o $DATASET_CUT.json.html 
