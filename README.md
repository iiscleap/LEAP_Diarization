## LEAP Diarization System for the Second DIHARD Challenge

This repository contains the code of the Interspeech 2019 paper titled as above which elucidates the DIHARD II submission of LEAP lab, Indian Institute of Science, Bengaluru.

The posterior scaled VB_diarization is inspired by 
- Kenny, P. Bayesian Analysis of Speaker Diarization with Eigenvoice Priors,
  Montreal, CRIM, May 2008.
-  [Variational-Bayes HMM](https://speech.fit.vutbr.cz/software/vb-diarization-eigenvoice-and-hmm-priors) 
 
#### Prerequisites

**1.** [Python](https://www.python.org/) >= 2.7

**2.** [dscore](https://github.com/nryant/dscore)

**3.** [DIHARD Baseline ](https://github.com/iiscleap/DIHARD_2019_baseline_alltracks)

#### Datasets and Models
**0.** We have used DIHARD Baseline (mentioned above) to generate features, GMM model and initial rttms.

**1.** The code is developed to test DIHARD II diarization but can be run for other datasets also 

**2.** The pre-trained model of diagonal ubm is present in modelFiles folder in h5 format. It is trained using 2048 GMM with MFCC features as mentioned in the paper and the Total variability is trained to get 400 dimensional speaker factor. Due to size contraints could not be uploaded but it is similar to T-matrix training.
In case of any help ask here.

#### Procedure to run code
**1.**  Download the github repo using link 
```bash
git clone https://github.com/iiscleap/LEAP_Diarization.git
```
**2.** Go inside scriptsvb folder
```bash
cd scriptsvb
```
**3.** Type following command for arguments help
```bash
python VB_main.py -h
```
**4.** Open file vbrun.sh and provide all the arguments present inside. Paths / name of folders are compulsory
```bash
vi vbrun.sh
```

#### Note
Only fold_local argument need absolute full path to your scripts folder. All other are relative paths, so only need the name of folders.
#### Outputs
**1.** Running the code generates one rttm_generated folder outside the scriptsvb folder. 

**2.** Inside it, there will be sub-folders created with name as the parameters value provided. It contains rttms of the input list provided along with score folder.

**3.** Score folder contains scores (DER, JER etc.) generated for each file provided in filelist using dscore/scrore.py
