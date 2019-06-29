# This script does following
#1. converts kaldi dgmm into text dgmm
#2. Reads text dgmm and save it into h5 using matlab code
#3. train T matrix 


. ./cmd.sh
. ./path.sh
set -e

dubm_in=/home/rakeshp/Diarization/kaldi-master/egs/callhome_diarization/v1/exp_b/extractor_c2048_i128/final.dubm
modeloutpath=`pwd`/../modelFiles
traindata=/home/rakeshp/Diarization/kaldi-master/egs/callhome_diarization/v1/data_b
#matscp_to_npy=/home/harshav/work/dihard_2018/v1/kaldi_io_for_python/read_scp_write_npy_embeddings.py
mkdir -p $modeloutpath

stage=4

. parse_options.sh || exit 1;

if [ $# != 0 -o "$dubm_in" = "default" ]; then
  echo "Usage: $0 --dubm_in <dubm model> --modeloutpath <path of dubm model h5>"
  exit 1;
fi

if [ $stage -le 1 ]; then
  echo "stage 1 : convert dubm to text "

  gmm-global-copy --binary=false $dubm_in $modeloutpath/callhome_dubm.txt

fi

if [ $stage -le 2 ]; then
  echo "stage 2: convert dubm.txt to h5 "

  /home/prachis/miniconda3/bin/python gen_dubmh5.py $modeloutpath/dubm.txt $modeloutpath/dubm.h5


fi


if [ $stage -le 3 ]; then
  echo "stage 3: Train t-matrix using training features"

  /home/prachis/miniconda3/bin/python Tmat_training/Tmatrix_training.py diarization_40_mfcc.cfg -o logs/log_tmat_train.txt -s 1 

fi