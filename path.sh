#Provide kaldi path in kaldi root
export KALDI_ROOT=`pwd`/../../.. 
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$KALDI_ROOT/tools/sctk/bin:$PWD:/usr/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
#export LD_LIBRARY_PATH=/opt/intel/bin:/opt/intel/mkl/lib/intel64_lin:/state/partition1/softwares/miniconda/pkgs/mkl-2018.0.1-h19d6760_4/lib:$LD_LIBRARY_PATH
