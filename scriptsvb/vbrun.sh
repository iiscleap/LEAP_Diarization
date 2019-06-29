#!bin/bash
echo "fold_local, initrttm_folder_name, feats_folder_name, filelist, sad_marks are compulsory arguments"

python VB_main.py \
--fold_local /home/data1/prachis/Dihard_2019/VB_HMM_xvec_Init_2019/scriptsvb/ \
--initrttm_folder_name initXvec_512 \
--feats_folder_name dihard_2019_dev_cmn \
--filelist  /home/data1/prachis/Dihard_2019/VB_HMM_xvec_Init_2019/batchfiles/dihardDev_aa \
--sad_marks labels_dihard_dev_2019_speech \
--featswithsad True \
--beta 24 \
--loopprob 0.5 \
--ivec-dim 400 \
--mindur 1 \
--downsamp 20 \
--dubmh5 final_vox_dubm_r.h5 --mytvmat tvmat.h5