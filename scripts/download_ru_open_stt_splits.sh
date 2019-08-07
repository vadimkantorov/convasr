SPLITS_DIR=data

mkdir -p "$SPLITS_DIR"
for f in calls_val.csv clean_train.csv clean_val.csv mixed_small.csv mixed_train.csv mixed_val.csv; do
    wget -q https://github.com/vadimkantorov/open_stt_splits/releases/download/with_excluded_by_cer/$f -P "$SPLITS_DIR"
done
