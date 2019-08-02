DATASET_LIBRISPEECH=$1

wget \
  http://www.openslr.org/resources/12/dev-clean.tar.gz \
  http://www.openslr.org/resources/12/dev-other.tar.gz \
  http://www.openslr.org/resources/12/test-clean.tar.gz \
  http://www.openslr.org/resources/12/test-other.tar.gz \
  http://www.openslr.org/resources/12/train-clean-100.tar.gz \
  http://www.openslr.org/resources/12/train-clean-360.tar.gz \
  http://www.openslr.org/resources/12/train-other-500.tar.gz \
  -P "$DATASET_LIBRISPEECH"
