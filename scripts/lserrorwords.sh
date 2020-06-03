#python3 tools.py lserrorwords --freq data/term2freq.dat --comment data/error_words_comment.csv -i data/speechcore/seq2seq_decoder/reports/not_finetuned_domain_set/Greedy.json --sortasc freq

#python3 tools.py processcomments --comment data/comments.csv -i data/speechcore/seq2seq_decoder/reports/not_finetuned_domain_set/Greedy.json -o data/filtered.json

#python3 vis.py errors data/speechcore/seq2seq_decoder/reports/not_finetuned_domain_set/Greedy.json data/speechcore/seq2seq_decoder/reports/not_finetuned_domain_set/Noise_finetuned.json --include data/speechcore/filtered.json --sortdesc cer -o data/vis.html
