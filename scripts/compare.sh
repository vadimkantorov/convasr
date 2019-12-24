OURS=data/speechcore/openstt_bad_model.json
THEIRS=data/speechcore/best_domain_model.json

AUDIOFILENAME=data/speechcore/openstt_bad_model.json.subset_cer_min0.5_maxNone.txt
python3 vis.py errors $OURS --theirs $THEIRS --audio-file-name $AUDIOFILENAME --audio

AUDIOFILENAME=data/speechcore/openstt_bad_model.json.subset_cer_minNone_max0.3.txt
python3 vis.py errors $OURS --theirs $THEIRS --audio-file-name $AUDIOFILENAME --audio

AUDIOFILENAME=data/speechcore/openstt_bad_model.json.subset_cer_min0.3_max0.5.txt
python3 vis.py errors $OURS --theirs $THEIRS --audio-file-name $AUDIOFILENAME --audio
