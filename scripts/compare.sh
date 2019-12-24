AUDIOFILENAME=data/speechcore/valset_mer_control/transcripts_valset_by_rec22122019.csv_GreedyDecoder.json.subset_cer_min0.5_maxNone.txt

OURS=data/transcripts_valset_by_rec22122019.csv_GreedyDecoder.json

THEIRS=

python3 vis.py errors $OURS --theirs $THEIRS --audio-file-name $AUDIOFILENAME
