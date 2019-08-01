DATASET_OPEN_STT=$1

aria2c http://academictorrents.com/download/a12a08b39cf3626407e10e01126cf27c198446c2.torrent -d "$DATASET_OPEN_STT" --seed-time=0

cd "$DATASET_OPEN_STT/ru_open_stt_wav"

for f in asr_calls_2_val.tar.gz buriy_audiobooks_2_val.tar.gz public_youtube700_val.tar.gz asr_public_stories_1.tar.gz asr_public_stories_2.tar.gz public_lecture_1.tar.gz public_series_1.tar.gz public_youtube1120.tar.gz radio_2.tar.gz ru_ru.tar.gz public_youtube1120_hq.tar.gz russian_single.tar.gz voxforge_ru.tar.gz asr_public_phone_calls_1.tar.gz; do
  tar -xf $f
  rm $f
done

for f in audiobooks_2.tar.gz_ public_youtube700.tar.gz_ asr_public_phone_calls_2.tar.gz_; do
  cat $f* > tmp.tar.gz
  rm $f*
  tar -xf tmp.tar.gz
  rm tmp.tar.gz
done
