# echomsk dataset
```shell
python3 echomsk.py ../echomsk/personalno_20000101_20191231.txt.json.gz --sample 10 --name sample10

wget -i sample10/sample10.txt -P sample10
```

# Download the exclude files
The gzipped versions are available at https://github.com/vadimkantorov/convasr/releases/tag/openstt_benchmark_files_backup

```shell
wget https://github.com/vadimkantorov/convasr/releases/download/openstt_benchmark_files_backup/benchmark_v05_public.csv.gz https://github.com/vadimkantorov/convasr/releases/download/openstt_benchmark_files_backup/clean_thresholds_cer.json https://github.com/vadimkantorov/convasr/releases/download/openstt_benchmark_files_backup/exclude_df_youtube_1120.csv.gz https://github.com/vadimkantorov/convasr/releases/download/openstt_benchmark_files_backup/public_exclude_file_v5.csv.gz https://github.com/vadimkantorov/convasr/releases/download/openstt_benchmark_files_backup/public_meta_data_v04_fx.csv.gz

python3 openstt.py
```

# Original exclude files
```shell
wget https://ru-open-stt.ams3.digitaloceanspaces.com/public_meta_data_v04_fx.csv

wget https://github.com/snakers4/open_stt/files/3348311/public_exclude_file_v5.zip
wget https://github.com/snakers4/open_stt/files/3348314/public_exclude_file_v5.z01.zip
wget https://github.com/snakers4/open_stt/files/3348312/public_exclude_file_v5.z02.zip
wget https://github.com/snakers4/open_stt/files/3348313/public_exclude_file_v5.z03.zip

cat public_exclude_file_v5.z01.zip public_exclude_file_v5.z02.zip public_exclude_file_v5.z03.zip public_exclude_file_v5.zip > public_exclude_file_v5_.zip
unzip public_exclude_file_v5_.zip

wget https://github.com/snakers4/open_stt/files/3386441/exclude_df_youtube_1120.zip
unzip exclude_df_youtube_1120.zip

wget https://ru-open-stt.ams3.digitaloceanspaces.com/benchmark_v05_public.csv.zip
zcat benchmark_v05_public.csv.zip > benchmark_v05_public.csv

rm public_exclude_file_v5.z01.zip public_exclude_file_v5.z02.zip public_exclude_file_v5.z03.zip public_exclude_file_v5.zip public_exclude_file_v5_.zip exclude_df_youtube_1120.zip benchmark_v05_public.csv.zip

gzip *.csv
```

# Download the dataset
```shell
apt-get update && apt-get install -y aria2
aria2c https://academictorrents.com/download/a7929f1d8108a2a6ba2785f67d722423f088e6ba.torrent --seed-time=0

cd ru_open_stt_wav

for f in asr_calls_2_val.tar.gz buriy_audiobooks_2_val.tar.gz public_youtube700_val.tar.gz asr_public_stories_1.tar.gz asr_public_stories_2.tar.gz public_lecture_1.tar.gz public_series_1.tar.gz public_youtube1120.tar.gz ru_ru.tar.gz public_youtube1120_hq.tar.gz russian_single.tar.gz voxforge_ru.tar.gz asr_public_phone_calls_1.tar.gz; do
  tar -xf $f
  rm $f
done

for f in audiobooks_2.tar.gz_ public_youtube700.tar.gz_ asr_public_phone_calls_2.tar.gz_ tts_russian_addresses_rhvoice_4voices.tar_; do
  cat $f* > tmp.tar.gz
  rm $f*
  tar -xf tmp.tar.gz
  rm tmp.tar.gz
done
```

```
splits/clean_train.csv          utterances: 26 K  hours: 48
splits/clean_val.csv            utterances: 1 K  hours: 2
splits/mixed_train.csv          utterances: 2020 K  hours: 2686
splits/mixed_val.csv            utterances: 15 K  hours: 9
splits/mixed_small.csv          utterances: 202 K  hours: 268
splits/calls_val.csv            utterances: 12 K  hours: 7


splits/clean_train.csv          utterances: 26 K  hours: 48
splits/clean_val.csv            utterances: 1 K  hours: 2
splits/mixed_train.csv          utterances: 2046 K  hours: 2698
splits/mixed_val.csv            utterances: 28 K  hours: 17

splits/clean_train.csv          utterances: 28 K  hours: 50
splits/clean_val.csv            utterances: 1 K  hours: 2
splits/mixed_train.csv          utterances: 2638 K  hours: 3369
splits/mixed_val.csv            utterances: 28 K  hours: 17

splits/clean_train.csv                          utterances: 37 K   hours: 58
splits/clean_val.csv                            utterances: 1 K    hours: 3
splits/addresses_train_mini.csv                 utterances: 37 K   hours: 16
splits/addresses_val_mini.csv                   utterances: 1 K    hours: 0
splits/audiobooks_train_mini1.csv               utterances: 37 K   hours: 49
splits/audiobooks_val_mini1.csv                 utterances: 1 K    hours: 2
splits/audiobooks_train_mini2.csv               utterances: 37 K   hours: 49
splits/audiobooks_val_mini2.csv                 utterances: 1 K    hours: 2
splits/audiobooks_train_mini3.csv               utterances: 37 K   hours: 49
splits/audiobooks_val_mini3.csv                 utterances: 1 K    hours: 2
splits/audiobooks_train.csv                     utterances: 1010 K hours: 1352
splits/audiobooks_train_medium.csv              utterances: 500 K  hours: 669
splits/audiobooks_train_mini.csv                utterances: 111 K  hours: 148
splits/audiobooks_val.csv                       utterances: 5 K    hours: 7
splits/mixed_train.csv                          utterances: 111 K  hours: 124
splits/mixed_val.csv                            utterances: 5 K    hours: 6
```
