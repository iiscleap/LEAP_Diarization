#!/usr/bin/env python

# Copyright 2013-2017 Lukas Burget (burget@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Revision History
#   L. Burget   16/07/13 01:00AM - original version

import numpy as np
from VB_diarization import VB_diarization, precalculate_VtiEV, frame_labels2posterior_mx
import htkmfc as htk
import glob
import math
import sys
import subprocess
import os
import matplotlib.pyplot as plt
import operator
#from functions import extract_mfcc
import os
import multiprocessing as mp
from lib.utils import h5read
import sys
from arguments import parse_arguments as args

param,path=args()

print("Arguments passed: ",sys.argv)

print('Dimensions:\n Features- D, Number of frames- N \n features shape: D X N  \n GMM:\n mixtures- C \n Weights: 1 X C ,Means shape: D X C, Variances: D X C \n T-matrix: R X CD ')
featswithsad = param.featswithsad
loopprob = param.loopprob
stat = param.beta
vec_dim = param.ivec_dim
vstat = 'hardasgn'

dihard_2019=path.feats_folder_name
SAD=path.sad_marks
initrttm_folder=path.initrttm_folder_name
fold_local= path.fold_local #'/home/data1/prachis/Dihard_2019/VB_HMM_xvec_Init_2019/scripts/'
def convert_to_rttm(ind,filename,rttm_newfile):


   rttm_new=open(rttm_newfile,'w')
   start=0
   change=[ i for i, (x, y) in enumerate(zip(ind[:-1],ind[1:])) if x!=y]
   change=np.array(change)+1
   change=np.append(change,len(ind))
   chold=0.0
   if len(change[:-1])==0:
          ch1=np.round(len(ind)/100.0-chold,3)
          
          begin='SPEAKER '+filename+' 1 '
          mid=str(chold)+' '+str(ch1)
          term=' <NA> <NA> ' + str(ind[-1])+ ' <NA> <NA>\n'
          total=begin+mid+term
          #print(total)
          #chold=ch/100.0
          if ind[-1]==99:
              print('silence')
          else:
              rttm_new.write(total)
   else:
       for ch in change:          
              ch1=np.round(ch/100.0-chold,3)
              
              begin='SPEAKER '+filename+' 1 '
              mid=str(chold)+' '+str(ch1)
              term=' <NA> <NA> ' + str(ind[ch-1])+ ' <NA> <NA>\n'
              total=begin+mid+term
              #print(total)
              chold=ch/100.0
              if ind[ch-1]==99:
                continue
              rttm_new.write(total)
              #chs=np.round(ch/100.0,3)+chs
   
      #start=end
   rttm_new.close()
   
def compute_score(filename,rttm_gndfile,rttm_newfile,outpath):
  scorecode='score.py -r '
  cmd='/home/prachis/miniconda3/bin/python '+ fold_local + '/dscore-master/' + scorecode + rttm_gndfile + ' -s ' + rttm_newfile + ' > ' + outpath + '.txt'  
  os.system(cmd)

 

def rttm2labels(rttm_sysfile):
  ind=[]
  silencedictnew={}
  rttm=np.genfromtxt(rttm_sysfile,dtype='str')

  if rttm.ndim==1:
      rttm=rttm.reshape(1,-1)

  rttm_n=(rttm[:,[3,4]].astype(float)*100).astype(int)
  rttm_label=[int(x.split('r')[-1]) for x in rttm[:,7]]
  end=rttm_n[:,0]+rttm_n[:,1]
  if rttm_n[0,0]==0:
    b=[rttm_label[0]]*rttm_n[0,1]
    ind.extend(b)
  else:
    b=[99]*(rttm_n[0,0])
    ind.extend(b)
    silencedictnew[0]=rttm_n[0,0]
    b=[rttm_label[0]]*rttm_n[0,1]
    ind.extend(b)
    
  endprev=end[0]
  
  k=1
  count=0
  for i in range(1,len(rttm_n)):
    start=rttm_n[i,0]
    diff=start-endprev
    if diff>0:
      #silencedictnew[endprev]=start
      ep=len(ind)
      b=[99]*(diff)
      ind.extend(b)
      st=len(ind)-1
      #print(ep,st)
      silencedictnew[ep]=ep+diff
    b=[rttm_label[i]]*rttm_n[i,1]
    ind.extend(b)
    endprev=end[i]    
  return np.array(ind),silencedictnew

def rttm2labels_new(rttm_sysfile):
  silencedictnew={}
  rttm=np.genfromtxt(rttm_sysfile,dtype='str')
  #print(rttm)
  if rttm.ndim==1:
      rttm=rttm.reshape(1,-1)
  #rttm_n=np.round((rttm[:,[3,4]].astype(float)*100)).astype(int)
  rttm_n=(rttm[:,[3,4]].astype(float)*100).astype(int)
  #rttm_label=rttm[:,7].astype(int)

  rttm_label=[int(x.split('r')[-1]) for x in rttm[:,7]]
  end_seg=rttm_n[:,0]+rttm_n[:,1]

  ind = np.ones((end_seg[-1],1))*99
  end_prev=0
  for i in range(len(end_seg)):
    start = rttm_n[i,0]
    end = end_seg[i]
    lab=rttm_label[i]
    ind[start:end]=lab
    if start-end_prev>0:
      silencedictnew[end_prev]=start
    end_prev=end
  ind = ind.astype(int)
  return ind,silencedictnew





def load_model_param():
  model_fold=fold_local + '../modelFiles/'
  dubm=path.dubmh5
  tmat=path.mytvmat
  ubm=model_fold+dubm
  Tmat=model_fold +tmat
  m,var,w=h5read(ubm,['means','variances','weights'])
  m=m.T  # C X D  
  w=w.T  #C X 1
  iE=1/var
  iE=iE.T # C X D
  V=h5read(Tmat,['T'])
  V=np.array(V[0]).reshape(-1,*m.shape) # R X C X D
  return m,iE,w,V


filelist=path.filelist
#files=np.genfromtxt(fold_local + '../batchfiles/' + filelist , dtype='str')
files = np.genfromtxt(filelist , dtype='str')


def generate_rttm(f,mindur,downsamp):
  print(f)
  filename=f#'DH_0160'
  initialize=1

  feats_file =fold_local + '/../feats/' + dihard_2019+'/'+filename+'.npy'

  rttm_gndfile=fold_local+'/../rttm_ground/'+filename+'.rttm'
  rttm_newfold=fold_local+'/../rttm_generated/{}frame_{}D_downsample_{}_loop_{}_statScale_{}_{}/'.format(mindur,vec_dim,downsamp,loopprob,stat)

  rttm_newfile=rttm_newfold+filename+'.rttm'

  speech_labels= (np.loadtxt(fold_local+'/../lists/'+SAD+'/'+filename)).astype(int) 
  change_points=[ i for i, (x, y) in enumerate(zip(speech_labels[:-1],speech_labels[1:])) if x!=y]
  change_points =np.array(change_points) +1
  if not os.path.isdir(rttm_newfold):
    os.makedirs(rttm_newfold)

  rttm_sysfile=fold_local+'/../'+initrttm_folder+'/'+filename+'.rttm'
  if not os.path.isfile(rttm_newfile):
    ref_initial,silencedictnew=rttm2labels_new(rttm_sysfile)
    outpath_fold=rttm_newfold+'/score/'
    outpath=outpath_fold+filename

    if not os.path.isdir(outpath_fold):
        os.makedirs(outpath_fold)
    
    
  # read UBM and iXtractor (total variability) subspace
  
  #feature extraction and label generation using rttm 

    X=np.load(feats_file) #D X N
    X=X.T  #N X D
    print(filename,X.shape)

    if len(ref_initial)>len(speech_labels):
      diff=len(ref_initial)-len(speech_labels)
      ref_initial=ref_initial[:-diff]
    elif len(ref_initial)<len(speech_labels) :
      diff=len(speech_labels)-len(ref_initial)
      b=[ref_initial[-1]]*diff
      ref_initial=list(ref_initial)
      ref_initial.extend(b)
      ref_initial=np.array(ref_initial)

    feats_speech=np.where(speech_labels<99)[0]

    if not featswithsad:
      X=X[feats_speech]

    index_speech=np.where(ref_initial<99)[0]
    print(ref_initial.shape, len(index_speech),len(feats_speech))
    

    new_ind = ref_initial[feats_speech]
    new_sil=np.where(new_ind==99)[0]
    
    new_ind[new_sil] = new_ind[new_sil-1]
    new_sil2=np.where(new_ind==99)[0]
    new_ind[new_sil2] = new_ind[new_sil2+1]
    new_ind=new_ind-np.min(new_ind)
    print('X dim {0} index dim {1}'.format(X.shape,len(new_ind)))
    if len(new_ind)>len(X):
      diff=len(new_ind)-len(X)
      new_ind=new_ind[:-diff]
    elif len(new_ind)<len(X) :
      diff=len(X)-len(new_ind)
      b=[new_ind[-1]]*diff
      new_ind=list(new_ind)
      new_ind.extend(b)
      new_ind=np.array(new_ind)
      
    m,iE,w,V=load_model_param()
    print('final_index',len(new_ind))
    ref=None
    VtiEV = precalculate_VtiEV(V, iE)
    q = None
    
    maxSpeakers=len(np.unique(new_ind))
    new_ind = new_ind[:,0]
    q = frame_labels2posterior_mx(new_ind) # initialize from the PLDA-AHC
    qold=q
    # for soft assignment
    # if maxSpeakers!=1:
    #   minprob=0.7
    #   q[qold==1.0]=minprob
      
    #   for ir,row in enumerate(qold):
    #       for ie,ele in enumerate(row):
    #           if ele==0.0:
    #               q[ir,ie]=(1-minprob)/(maxSpeakers-1)
                
    
    
    sp=None
    qold=q
   
    
# #    # runing with one frame resolution
    q, sp, L ,opt_path= VB_diarization(X, m, iE, w, V,filename,fold_local, sp=sp,q=q, maxSpeakers=maxSpeakers, maxIters=1, VtiEV=VtiEV, \
                                      downsample=downsamp, alphaQInit=100.0, sparsityThr=1e-6, epsilon=1e-6, minDur=mindur, \
                                      loopProb=loopprob,statScale=stat, llScale=1.0, ref=ref, plot=False)
 #    

    orgsilencedict={} 
    change_points =np.append(change_points,len(speech_labels))
    ch_prev=0
    if len(change_points)>1:
      for ch in change_points:

          if speech_labels[ch-1]==99:
            orgsilencedict[ch_prev]=ch
          ch_prev=ch
    else:
      orgsilencedict[change_points[-1]]=change_points[-1]

    print('q',np.sum(q,axis=1))
    q1=list(q)
    ind=[]
    count=0
    diff = 0
    for i in range(len(q1)):
        index, value = max(enumerate(q1[i]), key=operator.itemgetter(1))
        for st,end in orgsilencedict.items():
          st=int(st)
          end=int(end)          
          if count == st:
            diff = (end)-(st)            
            sil=[99]*diff           
            ind.extend(sil)
            count=end            
            break
        
        ind.append(index)
        count=count+1
        #print(count)

    convert_to_rttm(ind,filename,rttm_newfile)
    compute_score(filename,rttm_gndfile,rttm_newfile,outpath)

  
def gen_rttmlist(rttmlist):

    m=int(param.mindur) #mindur[i]
    d=int(param.downsamp) #downsamp[j]

    for f in rttmlist:
        #f='DH_0152'
        generate_rttm(f,m,d)
        #break
    
    
print('file',files)
files=np.array(files)
print(files)
gen_rttmlist(files)
