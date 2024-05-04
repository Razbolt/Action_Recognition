# Action Recognition Classification Project
 This project is Action Recognition Classfication build upon UCF101 dataset. 
 It takes frames of images(videos) and classify them based on its label.

## Project Structure

- `train.py`: This script is used for training the model. It loads the data and trains it based on the given model.
- `models.py`: This file contains different models with their own architectures.
- `test.py`: This script is used for testing the model. It loads the test data, trains it from the models in the models folder, and includes results as a confusion matrix and classification report in the `test_results` folder.
- `utils.py`: This script contains utility functions, including a function to activate the .yaml configuration file.
- `logger.py`: This script is used for logging and is helpful for tracking the training process.

## Installation

This project requires Python 3.8 or later. If you don't have Python installed, you can find it [here](https://www.python.org/downloads/).

```bash
python3 -m venv env

Activate the virtual environment 
On WIndows, run:
.\env\Scripts\activate

On Unix or MacOS, run:
source env/bin/activate

Then, install the necessary dependencies by running:
pip install -r requirements.txt
```
