T=data/speechcore/seq2seq_decoder/reports/not_finetuned_domain_set/Greedy.json

python3 tools.py lserrorwords --freq data/term2freq.dat --comment-path data/error_words_comment.csv -i $T --sortasc freq -o data/label.json --comment-filter typo

python3 vis.py label -i $T --info data/label.json -o data/label.json.html --prefix typo

#python3 tools.py processcomments --comment data/error_words_comment.csv -i data/speechcore/seq2seq_decoder/reports/not_finetuned_domain_set/Greedy.json -o data/filtered.json

#python3 vis.py errors data/speechcore/seq2seq_decoder/reports/not_finetuned_domain_set/Greedy.json data/speechcore/seq2seq_decoder/reports/not_finetuned_domain_set/Noise_finetuned.json --strip-audio-path-prefix /data/ --include data/speechcore/filtered.json --sortdesc cer --cer -0.5 --topk 500 --audio -o data/visaudio500.html
