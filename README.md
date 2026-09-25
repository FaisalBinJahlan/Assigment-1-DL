# ARTI 502 - Deep Learning Assignment 1

Training and testing neural networks on the MNIST handwritten digit dataset using PyTorch.

## Project Overview

The purpose of this assignment is to build a basic neural network, train it on MNIST, evaluate its performance, and study how different hyperparameters and network architectures affect the results.

The project starts with a simple fully connected network as required in the assignment. Different learning rates, momentum values, hidden layers, and numbers of neurons are then compared. A CNN experiment was also included to improve image classification accuracy.

## Dataset

MNIST contains grayscale images of handwritten digits from 0 to 9. Each image has a size of 28 x 28 pixels.

The dataset was divided into:

- 50,000 training images
- 10,000 validation images
- 10,000 testing images

The validation set was used for model selection and parameter tuning. The test set was kept separate for the final evaluation.

## Baseline Network

The baseline model contains:

- 784 input features after flattening each image
- One hidden layer with 128 neurons
- ReLU activation
- An output layer with 10 classes
- CrossEntropyLoss
- SGD optimizer with learning rate and momentum

The baseline model achieved a test accuracy of **97.64%**.

## Parameter and Architecture Tuning

The following settings were compared:

- Learning rates of 0.01 and 0.03
- Momentum values of 0.0 and 0.9
- Hidden layers with 128 and 256 neurons
- A deeper network with hidden layers of 512 and 256 neurons

The best fully connected configuration used two hidden layers with 512 and 256 neurons, a learning rate of 0.03, and momentum of 0.9.

## Additional CNN Experiment

A convolutional neural network was trained as an additional architecture experiment. Data augmentation, normalization, dropout, batch normalization, and learning-rate scheduling were used during training.

Two CNN models were trained using different random seeds. Their outputs were combined using weights selected from the validation results:

- CNN 1 weight: 0.3
- CNN 2 weight: 0.7

## Final Results

| Model | Evaluation Set | Accuracy |
|---|---|---:|
| Baseline neural network | Test | 97.64% |
| Best tuned fully connected model | Validation | 97.19% |
| CNN 1 | Validation | 99.47% |
| CNN 2 | Validation | 99.43% |
| Weighted CNN ensemble | Test | **99.63%** |

The final ensemble correctly classified **9,963 out of 10,000** test images, with a test loss of **0.0107**.

## Project Files

- `Assignment-1-DL.ipynb` - Complete notebook with code, explanations, outputs, and plots
- `assignment_1_dl.py` - Exported Python source code
- `requirements.txt` - Required Python packages
- `.gitignore` - Files excluded from the repository

## Running the Project

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install -r requirements.txt
```

Open `Assignment-1-DL.ipynb` in VS Code, select the `.venv` Python environment as the notebook kernel, and run the cells in order.

The dataset is downloaded automatically when the notebook is executed for the first time. CNN training may take several minutes when using the CPU.
