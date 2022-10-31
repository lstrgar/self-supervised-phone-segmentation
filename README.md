# Phoneme Segmentation Using Self-Supervised Speech Models

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
- If for some reason the link dissappears you may also try downloading the checkpoint directly from [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt)

***HuBERT***
- [Follow this link](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md#pre-trained-and-fine-tuned-asr-models)
- Select the download link where the "Model" column reads "HuBERT Base (~95M params)", the "Finetuning Dataset" column reads "No finetuning (Pretrained Model)", and the "Pretraining Data" is "	Librispeech 960 hr"
- For reference, we downloaded these models from this README page at git hash `4a7835b`
- If for some reason the link dissappears you may also try downloading the checkpoint directly from [here](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt)

### Obtain and Process TIMIT and/or Buckeye Speech Corpus

TIMIT is available [here](https://catalog.ldc.upenn.edu/LDC93S1) and Buckeye [here](https://buckeyecorpus.osu.edu/). 

Once the data has been obtained it must be stored in disk an a fashion that can be read by the provided dataloader, the core of which is borrowed from [Kreuk et al](https://github.com/felixkreuk/UnsupSeg). See the Data Structure section of this repo for specifics, or simply use the provided `utils/make_timit.py` and `utils/make_buckeye.py` to split and organize the data exactly how we did it. Both of these scripts we also credit to Kreuk et al, save a handful of minor changes. 

You can run `make_timit.py` and `make_buckeye.py` as follows:

`python utils/make_timit.py --inpath /path/to/original/timit --outpath /path/to/output/timit`

`python utils/make_buckeye.py --spkr --source /path/to/original/buckeye --target /path/to/output/buckeye --min_phonemes 20 --max_phonemes 50`

Note, here we do not provide the infrastructure to train these models using the pseudo-labels derived from a trained unsupervised model; however, the core implementation can be easily extended to train with alternate label supervision so long as the dataloader's interface remains unchanges. For those interested in training such a model, we would direct you to [Kreuk et al](https://github.com/felixkreuk/UnsupSeg), where a pretrained unsupervised model can be used to generate pseudo-labels for TIMIT. 

### Update Configuration YAML

The following fields will need to be updated to reflect local paths on your machine:

- timit_path
- buckeye_path
- base_ckpt_path

You may also want to experiment with the `num_workers` attribute depending on your hardware. 

### Training and Testing

To freeze the pre-trained model weights and train only a classifier readout model on TIMIT with a wav2vec2.0 backbone run the following
 
`python run.py data=timit lr=0.001 base_ckpt_path=/path/to/wav2vec2.0_ckpt mode=readout`

`data=timit` can easily be swapped for `data=buckeye` just as `base_ckpt_path=/path/to/wav2vec2.0_ckpt` can be swapped with `base_ckpt_path=/path/to/hubert_ckpt`. 

To finetune the whole pre-trained model and simply project final features with a linear readout run the you should set `lr=0.0001` and `mode=finetune`. Otherwise, the same swapping for TIMIT/Buckeye and wav2vec2.0/HuBERT applies. 

Invoking `run.py` will train a model from scratch for 50 epochs while printing training stats every 10 batches and running model validation every 50 batches. Print preferences can be changed in the config with attributes `print_interval` and `val_interval`. `epochs` can also be modified if desired.

During training models are saved to disk if they so-far demonstrate the best R-Value on the validation set. After training is complete, the best model is loaded from disk and tested with the testing set. Performance metrics in the harsh and lenient evaluation scheme are logged to standard out. 

Lastly, every invocation of `run.py` will create an output folder under `outputs/datestamp/{exp_name}_timestamp`, which is where model checkpoints are saved along with the whole runtime config and a run.log. Everything logged to standard output during training will also be logged to the run.log file. 

### Additional

This codebase assumes CUDA availability.

The config `seed` attribute can be changed to control random shuffling and initialization. 

`train_percent` indicates the fraction of the training set to use. Some may be interested in observing model / training data efficiency by sweeping over this attribute. Sweeps can be easily accomodated using hydra's multi-run command line option. For more see the hydra docs. 

As previously mentioned, several utility scripts in this codebase were borrowed from [Kreuk et al](https://github.com/felixkreuk/UnsupSeg). We thank these authors for their open source contributions. 
