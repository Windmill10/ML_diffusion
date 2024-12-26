# How to create conda environment
```conda env create -f env.yml --name <NAME_OF_ENV>```
We suggest using conda environment to avoid problems. Please create the environment using the above code.

# Datasets and Model Checkpoint
* [Google Drive](https://drive.google.com/drive/folders/1h3MHXx0NEIpVmY7uQcW1ETbO5NIqs29k)
You can find classified data folder here.

# Preprocessing


# Traning model
Please run
```python trainmultiple.py```
in the ```audio_diffusion_pytorch_trainer_main``` folder, which will automatically run the train program. 
You can adjust the species if needed.

# Generating
Run
```streamlit run ui_main.py``` 
to open the GUI interface. 
You can the choose the model you want to use there. Enter the seeds and steps and press the button to generate the sound. It might take a lttle time. 
Remind that though larger steps may increase the quality of the generated sound, it also make processing time longer. 

# Conditional Diffusion
* [Github](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)

# BirdNET Analyzer
* [Github](https://github.com/kahst/BirdNET-Analyzer)
