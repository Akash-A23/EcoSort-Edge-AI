# EcoSort: Edge-AI Vision Challenge

**Intel SIG Recruitment Challenge 2026**

EcoSort is a lightweight deep-learning image classification project designed to identify recyclable waste from 96 × 96 RGB images. The model classifies waste into six categories while adhering to strict model-size and parameter constraints for edge deployment.

## Problem Statement

The objective is to classify waste images into the following six classes:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The challenge requires a compact neural network exported in ONNX format, with no more than 350,000 parameters and a model size of at most 1.50 MB.

## Model Architecture

The project uses a custom lightweight convolutional neural network built using PyTorch and MobileNetV3-Small.

The architecture consists of:

1. **Feature extractor:** The first nine feature blocks of MobileNetV3-Small.
2. **Convolutional head:** A 1 × 1 convolution mapping 48 channels to 192 channels.
3. **Batch normalization:** Stabilizes intermediate feature activations.
4. **Activation:** Hard-Swish activation.
5. **Global average pooling:** Reduces spatial feature dimensions.
6. **Dropout:** Uses a dropout probability of 0.3 during training.
7. **Classifier:** A fully connected layer producing six class logits.

ImageNet-pretrained MobileNetV3-Small weights are used to initialize the feature extractor.

## Training Methodology

The model is trained using PyTorch with the following techniques:

- **Loss function:** Cross-entropy loss with label smoothing of 0.1.
- **Optimizer:** AdamW with configurable learning rate and weight decay of 0.01.
- **Learning-rate scheduler:** Cosine annealing.
- **Default training duration:** 30 epochs.
- **Default batch size:** 32.
- **Data augmentation:** Random resized cropping, horizontal and vertical flips, random rotation, and color jitter.
- **Validation:** Evaluates classification accuracy after every epoch and retains the weights with the best validation accuracy.

Images are resized or cropped to 96 × 96 pixels and normalized using ImageNet channel means and standard deviations.

## Project Structure

```text
EcoSort/
├── data/
│   └── public/
│       ├── train/
│       │   ├── cardboard/
│       │   ├── glass/
│       │   ├── metal/
│       │   ├── paper/
│       │   ├── plastic/
│       │   └── trash/
│       └── val/
│           ├── cardboard/
│           ├── glass/
│           ├── metal/
│           ├── paper/
│           ├── plastic/
│           └── trash/
├── train.py
├── model.onnx
└── README.md
```

The dataset directory structure must follow the class-folder layout expected by `torchvision.datasets.ImageFolder`.

## Installation

Install Python and the required libraries:

```bash
pip install torch torchvision onnx
```

Depending on the environment, install the appropriate PyTorch build for your hardware.

## Training

Run the training script from the project root:

```bash
python train.py
```

To customize the training configuration:

```bash
python train.py --data_dir data/public --epochs 30 --batch_size 32 --lr 0.001 --output model.onnx
```

The script automatically selects CUDA when available and otherwise uses the CPU. After training, it exports the best validation-performing model to ONNX format.

## ONNX Export

The model is exported using ONNX opset 17 with input and output names `input` and `output`, respectively. The batch dimension is dynamic.

The expected input tensor shape is:

```text
(batch_size, 3, 96, 96)
```

Input images must be converted to RGB tensors and normalized using the same ImageNet mean and standard deviation used during training.

The output contains six class logits in the order defined by the `CLASSES` list in `train.py`.

## Challenge Constraints

| Requirement | Limit |
|---|---:|
| Input resolution | 96 × 96 RGB |
| Number of classes | 6 |
| Model format | ONNX |
| Maximum parameters | 350,000 |
| Maximum model size | 1.50 MB |
| Required hidden-test accuracy | 75.0% |

The uploaded ONNX file is approximately 0.78 MiB, below the stated file-size limit. The architecture in the training script has approximately 201,278 parameters. ONNX validity and final hidden-test accuracy must be independently verified before submission.

## Reproducibility and Evaluation

The training script exposes command-line options for the dataset path, number of epochs, batch size, learning rate, and output filename. It reports validation accuracy for each epoch and exports the weights associated with the best validation result.

The final model should be checked using an ONNX validator and evaluated using the challenge's prescribed preprocessing and hidden-test evaluation procedure.

## Technologies Used

- Python
- PyTorch
- Torchvision
- MobileNetV3-Small
- ONNX

## Submission

The final submission artifact is `model.onnx`. Before submitting, verify that the exported model loads successfully, satisfies the parameter and file-size limits, produces the expected six-class output, and achieves the required hidden-test accuracy.
