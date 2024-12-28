# Bird Vocalization Generation Using Diffusion Models

This project generates bird sounds using a diffusion model. It is based on the [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch) repository. The original repository provides a framework for training and using diffusion models for audio generation in our project.

## Modifications and Additions

The following modifications and additions have been made to the original repository:

- Added crawler for downloading bird sounds from the eBird dataset.
- Added support for preprocessing bird sounds and classifying them using a pre-trained model.
- Automated training of the model for multiple bird species.
- Integrated the BirdNET Analyzer for bird sound analysis.
- Added new datasets and model checkpoints.
- Implemented a Streamlit interface for generating bird sounds.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Datasets and Model Checkpoint](#datasets-and-model-checkpoint)
- [Credits](#credits)

## Installation

Due to environmental conflicts, if the provided `requirements.txt` doesn't work, use the provided conda environments instead for each process.

To install requirements, run:
```bash
pip install -r requirements.txt
```
To create the conda environment, run:
```bash
conda env create -f cross_platform_env.yml --name <NAME_OF_ENV>
```

## Usage

### Preprocessing

#### Cropping and Classifying Audio Files

The preprocessing steps involve cropping audio files into segments and classifying them using a pre-trained model.

1. **Define the Root Folder of Bird Sounds**: Ensure that your raw audio data is placed in the `dataset/鳥種清單` directory.

2. **Define the Root Folder for Cropped Data**: The cropped audio data will be saved in the `dataset/cropped_data` directory.

3. **Run the Preprocessing Script**: Execute the `main.py` in preprocessing to crop and classify the audio files (it might take a while).

The script performs the following steps:

- Crops the audio files into 5-second segments with a stride of 1 second.
- Classifies the cropped audio segments using the `MIT/ast-finetuned-audioset-10-10-0.4593` model.
- Saves the classified audio segments in the `dataset/classified_data` directory.

#### Example Directory Structure
```
preprocessing/
├── dataset/
│   ├── 鳥種清單/
│   │   ├── bird1/
│   │   │   ├── audio1.mp3
│   │   │   ├── audio2.mp3
│   │   ├── bird2/
│   │   │   ├── audio1.mp3
│   │   │   ├── audio2.mp3
│   ├── cropped_data/
│   ├── classified_data/
├── preprocessing/
│   ├── main.py
│   ├── audioprocessing.py
```

### Training

#### Training the Model

To train the model for testing or single audio files or directories (for testing purposes), follow these steps:

1. **Prepare the Configuration File**: Create or modify a configuration file in the `exp` directory. You can use the provided `.yaml` files as templates. Ensure that the paths to your datasets and other parameters are correctly set.

2. **Run the Training Script**: Execute the training script with the desired configuration file. For example:
    ```bash
    python train.py exp=base_medium
    ```

3. **Monitor Training**: Training logs and checkpoints will be saved in the specified directories. You can monitor the training process using the Weights & Biases (wandb) logger if configured.

#### Example Command
```bash
python train.py exp=base_medium trainer.gpus=1
```

This command will start training using the `base_medium` configuration and utilize one GPU.

#### Resuming Training
If you need to resume training from a checkpoint, use the following command:
```bash
python train.py exp=base_medium +ckpt=/logs/ckpts/2022-08-17-01-22-18/'last.ckpt'
```

This will load the model from the specified checkpoint and continue training.

To train the model for the specific usage in this project, run the following command:
```bash
python trainmultiple.py 
```

#### Example Directory Structure
```
ML_diffusion/
├── dataset/
│   ├── cropped_data/
│   ├── classified_data/
├── training/
│   ├── exp/
│   │   ├── base_medium.yaml ...
│   ├── main/
│   ├── train.py
│   ├── trainmultiple.py
```

### Generation

#### Setting Up Model Checkpoints
In the `training` directory, rename a saved .ckpt of a species to its name. For example, `bird1.ckpt` for a bird species named `bird1`.

#### Generating Bird Sounds

To generate bird sounds using the trained model, follow these steps:

1. **Select Bird Species**: Use the Streamlit interface to select the bird species for which you want to generate sounds.

2. **Set Parameters**: Enter the seed number and the number of steps for the generation process.

3. **Generate Sound**: Click the "Generate" button to start the sound generation process. The generated sound will be displayed and played in the Streamlit interface. You can also download the generated sound. The BirdNET Analyzer will analyze the generated sound.

#### Example Command
```bash
streamlit run UI/ui_main.py
```

This command will start the Streamlit application, allowing you to interactively generate bird sounds.

#### Example Directory Structure
```
ML_diffusion/
├── training/
│   ├── ckpts/
│   │   ├── bird1.ckpt ...
├── UI/
│   ├── ui_main.py
│   ├── ui_pregenerate.py
```

## Datasets and Model Checkpoint

We provided the classified dataset, some pretrained model checkpoints and audio samples.
You can download them from the following link:
* [Google Drive](https://drive.google.com/drive/folders/1h3MHXx0NEIpVmY7uQcW1ETbO5NIqs29k)

## Credits

This project is based on the audio-diffusion-pytorch repository by archinetai. We would like to thank the original authors for their work and contributions.

### Original Repositories

* [audio-diffusion-pytorch](https://github.com/archinetai/audio-diffusion-pytorch)
* [audio-diffusion-pytorch-trainer](https://github.com/archinetai/audio-diffusion-pytorch-trainer)

### BirdNET Analyzer

* [BirdNET Analyzer](https://github.com/kahst/BirdNET-Analyzer)

### Conditional Diffusion

* [Conditional Diffusion MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)
