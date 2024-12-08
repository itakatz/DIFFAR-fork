### [This is my fork of DiffAR, for working with saxhophone audio / envelopes]

# DiffAR: Denoising Diffusion Autoregressive Model for Raw Speech Waveform Generation

Abstract: Diffusion models have recently been shown to be relevant for high-quality speech generation. Most work has been focused on generating spectrograms, and as such, they further require a subsequent model to convert the spectrogram to a waveform (i.e., a vocoder). This work proposes a diffusion probabilistic end-to-end model for generating a raw speech waveform. The proposed model is autoregressive, generating overlapping frames sequentially, where each frame is conditioned on a portion of the previously generated one. Hence, our model can effectively synthesize an unlimited speech duration while preserving high-fidelity synthesis and temporal coherence. We implemented the proposed model for unconditional and conditional speech generation, where the latter can be driven by an input sequence of phonemes, amplitudes, and pitch values. Working on the waveform directly has some empirical advantages. Specifically, it allows the creation of local acoustic behaviors, like vocal fry, which makes the overall waveform sounds more natural. Furthermore, the proposed diffusion model is stochastic and not deterministic; therefore, each inference generates a slightly different waveform variation, enabling abundance of valid realizations. Experiments show that the proposed model generates speech with superior quality compared with other state-of-the-art neural speech generation systems.

<img src="https://github.com/RBenita/DIFFAR/blob/main/docs/frame_explain_2.png?raw=true" width="300" height="200" align=right>

Visit our [demo page](https://rbenita.github.io/DIFFAR/) for audio samples.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.01381)

We provide a PyTorch implementation of our work.
#### This repository is currently under construction. ####

- synthesis examples are available in the "Examples" folder.
- An HTML file summarizing representative examples is available here: 
[Open html](https://github.com/RBenita/DIFFAR/blob/main/docs/index.html)



## DataSets ##
Currently, The supported dataset is:

LJSpeech: [GitHub Pages](https://keithito.com/LJ-Speech-Dataset/) a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.

## Preprocessing ##
Before training your model, make sure to have the following .json files:
1. _wav.json files:
```
- train_wav.json
- val_wav.json
```
An example can be found [here](https://github.com/RBenita/DIFFAR/blob/main/Demo_json_files/Demo_wav.json)


You can generate these files by running:
`python flder2json.py <wavs_directory> | <json_directory>`

2. _TextGride.json files:
```
- train_TextGrid.json
- val_TextGrid.json
```
An example can be found [here](https://github.com/RBenita/DIFFAR/blob/main/Demo_json_files/Demo_textgrid.json)


You can generate these files by running:
`python flder2json_txt.py <Textgrid_directory> | <json_directory>`

3. _Energy.json files:
```
- train_Energy.json
- val_Energy.json
```
An example can be found [here](https://github.com/RBenita/DIFFAR/blob/main/Demo_json_files/Demo_npy_energy.json)


You can generate these files by:
   * Make a folder with energy .npy  files using the function:  from_wav_file_to_npy_energy_file
   * Run `python flder2json_npy.py <Energy_directory> | <json_directory>`

## Training ##
Our implementation is Using [hydra](https://hydra.cc).

* Make sure you have updated the [conf.yaml](https://github.com/RBenita/DIFFAR/blob/main/conf/conf.yaml) file correctly. Mainly pay attention to the fields:
```
train_ds:
  json_wav: 
  json_TextGrids:
  json_npy_Energy:


valid_ds:
  json_wav: 
  json_TextGrids:
  json_npy_Energy: 
```

* run `__main__.py`
```
HYDRA_FULL_ERROR=1 python ./__main__.py
```
  

## Infernece ##
To synthesize your custom .wav files: 
1. Make sure to have a pre-trained model: `./models/DiffAR_200.pt`
2. Locate the .txt files under a folder named 'text_files' as follows:
```
|-- current_directory
|   |-- text_files
|   |   |-- file1.txt
|   |   |-- file2.txt
|   |   |-- file3.txt
```
   
3. run `python inference.py --main_directory <current_directory>`

A successful run should yield the following folder structure:

```
|-- current_directory
|   |-- text_files
|   |   |-- file1.txt
|   |   |-- file2.txt
|   |   |-- file3.txt
|   |-- predicted_energy_files
|   |   |-- file1.npy
|   |   |-- file2.npy
|   |   |-- file3.npy
|   |-- predicted_TextGrid_files
|   |   |-- file1.TextGrid
|   |   |-- file2.TextGrid
|   |   |-- file3.TextGrid
|   |-- generated_wavs
|   |   |-- file1.wav
|   |   |-- file2.wav
|   |   |-- file3.wav

```

## TODO
- [x] Training procedure for DiffAR 200
- [x] Inference Procedure for DiffAR 200
- [ ] Training and Infernce Procedure for DiffAR 1000
- [ ] Training and Infernce Procedure for DiffAR-E
- [ ] Training and Infernce Procedure for DiffAR+P


