# This script does following
#0. genrate mfccs for training T matrix. Refer DIHARD baseline for data prep
#1. train diagonal ubm
#2. train T matrix using diagonal ubm
#3. convert dioagonal ubm and T matrix to pickle format


. ./cmd.sh
. ./path.sh

set -e
stage=1
nnet_dir=exp/xvector_nnet_1a/

#Variational Bayes resegmentation options
VB_resegmentation=true
num_gauss=2048
ivec_dim=400

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  echo "Stage 1 NOWW end"
  #Make MFCCs for each dataset
  for name in train ; do
    steps/make_mfcc.sh --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd --max-jobs-run 40" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
  done

  #Compute the energy-based VAD for train
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
    data/train exp/make_vad $vaddir
  utils/fix_data_dir.sh data/train

  echo "VAD done !!!"

if [ $VB_resegmentation ]; then
  # Variational Bayes method for smoothing the Speaker segments at frame-level
  output_dir=exp/xvec_init_gauss_${num_gauss}_ivec_${ivec_dim}

if [ $stage -le 1 ]; then
 
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 20G --max-jobs-run 6" \
            --nj 10 --num-threads 4  --subsample 1 --delta-order 0 --apply-cmn true \
            data/train $num_gauss \
            exp/diag_ubm_gauss_${num_gauss}
fi

if [ $stage -le 2 ]; then
  utils/subset_data_dir.sh \
    --utt-list <(sort -n -k 2 data/train/utt2num_frames | tail -n 100000) \
    data/train data/train_100k
  # Train the i-vector extractor. The UBM is assumed to be diagonal.
  diarization/train_ivector_extractor_diag.sh --cmd "$train_cmd --mem 45G --max-jobs-run 20" \
                    --ivector-dim ${ivec_dim} \
                    --num-iters 5 \
                    --apply-cmn true \
                    --num-threads 1 --num-processes 1 --nj 10 \
                    exp/diag_ubm_gauss_${num_gauss}/final.dubm data/train_100k \
                    exp/extractor_gauss_${num_gauss}_ivec_${ivec_dim}
fi

if [ $stage -le 3 ]; then
  # Convert the Kaldi UBM and T-matrix model to pickle dictionary.
  mkdir -p $output_dir
  mkdir -p $output_dir/tmp
  mkdir -p $output_dir/log
  mkdir -p $output_dir/model

  # Dump the diagonal UBM model into text format.
  # "$train_cmd" $output_dir/log/convert_diag_ubm.log \
     gmm-global-copy --binary=false \
     exp/diag_ubm_gauss_${num_gauss}/final.dubm \
     $output_dir/tmp/dubm.tmp || exit 1;

  # Dump the ivector extractor model into text format.
  # This method is not currently supported by Kaldi,
  # so please use my kaldi.
  # "$train_cmd" $output_dir/log/convert_ie.log \
     ivector-extractor-copy --binary=false \
     exp/extractor_gauss_${num_gauss}_ivec_${ivec_dim}/final.ie \
     $output_dir/tmp/ie.tmp || exit 1;

  diarization/dump_model.py $output_dir/tmp/dubm.tmp $output_dir/model
  diarization/dump_model.py $output_dir/tmp/ie.tmp $output_dir/model
fi