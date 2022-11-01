# Phoneme Segmentation Using Self-Supervised Speech Models

## Paper

[Phoneme Segmentation Using Self-Supervised Speech Models](**LINK**)

Luke Strgar and David Harwath

Accepted to Spoken Language Technology (SLT) Workshop 2022

### Abstract

We apply transfer learning to the task of phoneme segmenta- tion and demonstrate the utility of representations learned in self-supervised pre-training for the task. Our model extends transformer-style encoders with strategically placed convolu- tions that manipulate features learned in pre-training. Using the TIMIT and Buckeye corpora we train and test the model in the supervised and unsupervised settings. The latter case is accomplished by furnishing a noisy label-set with the predic- tions of a separate model, it having been trained in an unsu- pervised fashion. Results indicate our model eclipses previ- ous state-of-the-art performance in both settings and on both datasets. Finally, following observations during published code review and attempts to reproduce past segmentation re- sults, we find a need to disambiguate the definition and im- plementation of widely-used evaluation metrics. We resolve this ambiguity by delineating two distinct evaluation schemes and describing their nuances.

### Cite

If you find our paper or this code useful, kindly consider citing our work:

```
**CITATION**
```

## Usage

### Clone This Repository

`git clone --recurse-submodules git@github.com:lstrgar/self-supervised-phone-segmentation.git phone-seg`

Now checkout the correct fairseq submodule branch: 

```
cd phone-seg/fairseq
git checkout lvs
cd ..
```

### Setup Environment
```
conda create --name phone-seg
conda activate phone-seg
conda install python=3.9.5
pip install ./fairseq
pip install -r requirements.txt
```

### Obtain Pre-trained Model Checkpoints
wav2vec2.0 and HuBERT checkpoints are available via fairseq at the following links. Download these models and place in a new folder titled `checkpoints`. 

***wav2vec2.0***
- [Follow this link](https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#pre-trained-models)
- Select the download link where the "Model" column reads "Wav2Vec 2.0 Base", the "Finetuning" column reads "No finetuning", and the "Dataset" is "Librispeech"
- For reference, we downloaded these models from this README page at git hash `edb25c6`
- If for some reason you are unable to obtain the model checkpoint according to the above steps you may also try to download it directly from [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)

***HuBERT***
- [Follow this link](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md#pre-trained-and-fine-tuned-asr-models)
- Select the download link where the "Model" column reads "HuBERT Base (~95M params)", the "Finetuning Dataset" column reads "No finetuning (Pretrained Model)", and the "Pretraining Data" is "	Librispeech 960 hr"
- For reference, we downloaded these models from this README page at git hash `4a7835b`
- If for some reason you are unable to obtain the model checkpoint according to the above steps you may also try to download it directly from [here](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)

### Obtain and Process TIMIT and/or Buckeye Speech Corpus

TIMIT is available [here](https://catalog.ldc.upenn.edu/LDC93S1) and Buckeye [here](https://buckeyecorpus.osu.edu/). 

Once the data has been obtained it must be stored in disk an a fashion that can be read by the provided dataloader, the core of which is borrowed from [Kreuk et al](https://github.com/felixkreuk/UnsupSeg). See the Data Structure section of this repo for specifics, or simply use the provided `utils/make_timit.py` and `utils/make_buckeye.py` to split and organize the data exactly how we did it. Both of these scripts we also credit to Kreuk et al, save several minor changes. 

`make_timit.py` simply renames and restructures the data directories such that paths to audio recordings and their corresponding segmentation label files are flattened. The standard TIMIT download looks something like this: 

```
- timit
    - data
        - TRAIN
            - DR1
                - FCJF0
                    - SA1.PHN
                    - SA1.TXT
                    - SA1.WAV
                    - SA1.WRD
                ...
            ...
        - TEST
    - PHONCODE.DOC
    - PROMPTS.TXT
    - README.DOC
    ...
```

Whereas the new directory created by `make_timit.py` will look like this:

```
- timit
    - train
        - DR1_SA1.phn
        - DR1_SA1.wav
        ...
    - train
```

Depending on the format of the TIMIT `.WAV` files, you may need to read the data and overwrite them using a standard encoding. We used [soundfile](https://pysoundfile.readthedocs.io/en/latest/) for these purposes. 

`make_buckeye.py` does more than just restructure the data -- the script splits long audio recordings into shorter segments that can be used for training. ***Before*** running `make_buckeye.py` the standard zip files must be unpacked such that the directory structure is preserved. Specifically, you may download each of the speaker zips independently and then recursively unzip each zip file to create a `buckeye` folder with the following structure: 

```
- buckeye
  - s01
    - s0101a.log
    - s0101a.phones
    - s0101a.txt
    - s0101a.wav
    - s0101a.words
    - ...
  - s02
  ...
```

You can run `make_timit.py` and `make_buckeye.py` as follows:

`python utils/make_timit.py --inpath /path/to/original/timit/data --outpath /path/to/output/timit`

`python utils/make_buckeye.py --spkr --source /path/to/original/buckeye --target /path/to/output/buckeye --min_phonemes 20 --max_phonemes 50`

You can expect the output of `make_timit.py` to look like this:

```
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.72s/it]
```

You can expect the output of `make_buckeye.py` to look like this:

```
  9%|████████████▉                                                                                                                                  | 23/255 [03:48<33:14,  8.60s/it]last phone end: 599.192625
len of wav: 568.739
skipping ../data/buckeye-raw/s19/s1901b.wav
 35%|██████████████████████████████████████████████████▍                                                                                            | 90/255 [14:03<20:33,  7.48s/it]last phone end: 570.44755
len of wav: 560.691
skipping ../data/buckeye-raw/s40/s4002a.wav
 42%|██████████████████████████████████████████████████████████▏                                                                                 | 106/255 [18:09<1:02:35, 25.20s/it]last phone end: 598.871688
len of wav: 574.51
skipping ../data/buckeye-raw/s29/s2902b.wav
 51%|████████████████████████████████████████████████████████████████████████▉                                                                     | 131/255 [23:54<32:21, 15.66s/it]last phone end: 327.297346
len of wav: 324.1
skipping ../data/buckeye-raw/s36/s3601b.wav
 59%|████████████████████████████████████████████████████████████████████████████████████                                                          | 151/255 [27:50<26:57, 15.56s/it]last phone end: 568.013937
len of wav: 564.609
skipping ../data/buckeye-raw/s27/s2702b.wav
 92%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊           | 235/255 [37:30<02:31,  7.56s/it]last phone end: 600.991312
len of wav: 600.0000625
skipping ../data/buckeye-raw/s10/s1003a.wav
 98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊  | 251/255 [38:15<00:05,  1.37s/it]loading ../data/buckeye-raw/s35/s3504a.wav failed!
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 255/255 [38:33<00:00,  9.07s/it]
10264 items
avg len: 32.117108339828526
min len: 22
max len: 50
avg sec: 2.696509640003895
min sec: 1.0300000000000296
max sec: 8.28861599999999
7.688048595833328 hours
```

Note, here we do not provide the infrastructure to train these models using the pseudo-labels derived from a trained unsupervised model; however, the core implementation can be easily extended to train with alternate label supervision so long as the dataloader's interface remains unchanged. For those interested in training such a model, we would direct you to [Kreuk et al](https://github.com/felixkreuk/UnsupSeg), where a pretrained unsupervised model can be used to generate pseudo-labels for TIMIT. When training with pseudo-labels, we advise computing precision and recall metrics of the pseudo-labels with respect to the ground truth data and setting the config attribute `pos_weight` to account precision/recall imbalance. `pos_weight` is multiplicative constant applied to loss computed at ground truth boundary locations. 

### Update Configuration YAML

The following fields will need to be updated to reflect local paths on your machine:

- `timit_path`
- `buckeye_path`
- `base_ckpt_path`

You may also want to experiment with the `num_workers` attribute depending on your hardware. 

### Training and Testing

To freeze the pre-trained model weights and train only a classifier readout model on TIMIT with a wav2vec2.0 backbone run the following
 
`python run.py data=timit lr=0.001 base_ckpt_path=/path/to/wav2vec2.0_ckpt mode=readout`

`data=timit` can easily be swapped for `data=buckeye` just as `base_ckpt_path=/path/to/wav2vec2.0_ckpt` can be swapped with `base_ckpt_path=/path/to/hubert_ckpt`. 

To finetune the whole pre-trained model and simply project final features with a linear readout run the you should set `lr=0.0001` and `mode=finetune`. Otherwise, the same swapping for TIMIT/Buckeye and wav2vec2.0/HuBERT applies. 

Invoking `run.py` will train a model from scratch for 50 epochs while printing training stats every 10 batches and running model validation every 50 batches. Print preferences can be changed in the config with attributes `print_interval` and `val_interval`. `epochs` can also be modified if desired.

During training models are saved to disk if they demonstrate the best seen R-Value on the validation set. After training is complete, the best model is loaded from disk and tested with the testing set. Performance metrics in the harsh and lenient evaluation scheme are logged to standard output. 

Every invocation of `run.py` will create an output folder under `outputs/datestamp/{exp_name}_timestamp`, which is where model checkpoints are saved along with the whole runtime config and a `run.log`. Everything logged to standard output during training will also be logged to the `run.log` file. 

### Additional

This codebase assumes CUDA availability.

The config `seed` attribute can be changed to control random shuffling and initialization. 

`train_percent` indicates the fraction of the training set to use. Some may be interested in measuring model / training data efficiency by sweeping over values of this attribute. Sweeps can be easily accommodated using hydra's multi-run command line option. For more see the hydra docs. 

As previously mentioned, several utility scripts in this codebase were borrowed from [Kreuk et al](https://github.com/felixkreuk/UnsupSeg). We thank these authors for their open source contributions. 
