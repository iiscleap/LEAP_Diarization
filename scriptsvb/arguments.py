# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Arguments for UISRNN."""

import argparse




def parse_arguments():
  """Parse arguments.

  Returns:
    A tuple of:
      param_args: parameters
      path_args: parameters
     
  """
  # parameters 
  param_parser = argparse.ArgumentParser(
      description='VB configurations.', add_help=False)

  param_parser.add_argument(
      '--featswithsad',
      default=True,
      type=bool,
      help='features contains only speech ,no silence (True/False)')

  param_parser.add_argument(
      '--beta',
      default=24,
      type=int,
      help='The beta factor')

  param_parser.add_argument(
      '--loopprob',
      default=0.5,
      type=float,
      help='loop probability')
  param_parser.add_argument(
      '--ivec-dim',
      default=400,
      type=int,
      help='i-vector dimension')
  param_parser.add_argument(
      '--mindur',
      default=1,
      type=int,
      help='minimum duration per speaker')
  param_parser.add_argument(
      '--downsamp',
      default=20,
      type=int,
      help='downsampling factor')

    # path configurations
  path_parser = argparse.ArgumentParser(
      description='path/folders of input features (must be inside feats), initialization rttm, full path of file_list,'
      ' name of folder containing sad labels, name of diagonal ubm and T matrix pre-trained(inside modelfiles)', add_help=False)


  path_parser.add_argument(
      '--fold_local',
      required=True,
      default='/home/data1/prachis/Dihard_2019/VB_HMM_xvec_Init_2019/scriptsvb/',
      type=str,
      help='Full path of scripts folder')
  path_parser.add_argument(
      '--initrttm_folder_name',
      required=True,
      default='initXvec_512',
      type=str,
      help='folder name where initial rttm are kept outside scriptsvb folder')
  
  path_parser.add_argument(
      '--feats_folder_name',
      required=True,
      default='dihard_2019_dev_cmn',
      type=str,
      help='name of featurs folder outside scriptsvb inside feats')

  path_parser.add_argument(
      '--filelist',
      required=True,
      default='dihardlist',
      type=str,
      help='list containing filename of audio')

  path_parser.add_argument(
      '--sad_marks',
      required=True,
      default='labels_dihard_dev_2019_speech',
      type=str,
      help='folder containg sad for audio files 1 for speech and 99 for silence')

  path_parser.add_argument(
      '--dubmh5',
      default='final_vox_dubm_r.h5',
      type=str,
      help='present in modelFiles folder, diagonal ubm model')

  path_parser.add_argument(
    '--mytvmat',
    default='tvmat.h5',
    type=str,
    help='present in modelFiles folder, T matrix model')
  # a super parser for sanity checks
  super_parser = argparse.ArgumentParser(
      parents=[param_parser, path_parser])

  # get arguments
  super_parser.parse_args()
  param_args, _ = param_parser.parse_known_args()
  path_args, _ = path_parser.parse_known_args()


  return (param_args, path_args)
