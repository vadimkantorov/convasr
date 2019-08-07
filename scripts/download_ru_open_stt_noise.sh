NOISE_DIR=data/ru_open_stt_noise

mkdir -p "$NOISE_DIR"
wget https://asr-noise.fra1.digitaloceanspaces.com/noises_df.feather -P "$NOISE_DIR"
wget https://asr-noise.fra1.digitaloceanspaces.com/asr_noises.tar.gz -P "$NOISE_DIR"

cd "$NOISE_DIR"
tar -xf asr_noises.tar.gz
rm asr_noises.tar.gz
