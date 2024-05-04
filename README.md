# Action Recognition Classification Project
 This project is Action Recognition Classfication build upon UCF101 dataset. 
 It takes frames of images(videos) and classify them based on its label.

## Project Structure

Files:
- `train.py`: This script is used for training the model. It loads the data and trains it based on the given model.
- `models.py`: This file contains different models with their own architectures.
- `test.py`: This script is used for testing the model. It loads the test data, trains it from the models in the models folder, and includes results as a confusion matrix and classification report in the `test_results` folder.
- `utils.py`: This script contains utility functions, including a function to activate the .yaml configuration file.
- `logger.py`: This script is used for logging and is helpful for tracking the training process.
- `C3_model.py`: This file contains the 3D CNN architecture from the paper.
- `config.yaml`: This is the configuration file, contains roots and annotation paths. You can change it and play around with batch size, and number of epochs, frame_per_clips and steps_betwen_clips

Folders:
- UCF-101 folder contains the dataset with 10 labels
- UCF-101-1 folder contains the dataset with 100 labels
- ucfTrainTestlist  folder contains the all of the annotation path 
- checkpoints folder checkpoins of the pre-trained models
- models folder has all of the trained models from our architectures

Unfortunately, due to limitation of git, these folder are ignored in .gitignore file, please take it from the zipped folder


## Installation

- This project requires Python 3.8 or later. If you don't have Python installed, you can find it [here](https://www.python.org/downloads/).
- Original dataset from this url: https://www.crcv.ucf.edu/data/UCF101.php
- For Resnet50 please download from the link and add into /checkpoints file: https://download.pytorch.org/models/resnet50-0676ba61.pth

To activate virtual environment:

```bash
python3 -m venv env
```
Activate the virtual environment 
On WIndows, run:
```bash
.\env\Scripts\activate
```
On Unix or MacOS, run:
```bash
source env/bin/activate
```

Then, install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

## Logger
At the beggining of the project it will weight&biases ```API_KEY```. PLease take it look at this following url:https://docs.wandb.ai/quickstart


## Training

In order to train the model first you need to follow the above instructions. Environment, Dataset and Checkpoints. 
After, you can take a closer look at config.yaml file and change some hyperparameters.
You can train the model in train.py with given .yaml **parameters(root, annotation path , frames per clip and steps between clips, number of epochs and batch size )**.
Run:
```bash 
python train.py
``` 
It will save the model in models folder at the end of the training.

## Testing

In order to test the model first its need to taken from the models folder and run the test.py with the same parameters from .yaml file.
To run:
```bash
python test.py

```
 At the end you can see the results in test/result folder with classification report, confusion matrix with accuracy score. Also its include everything in logger as alternative.


