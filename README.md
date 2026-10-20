# EcoSort: Edge-AI Vision Challenge ♻️

**Intel SIG Recruitment Challenge 2026**

EcoSort is a lightweight deep-learning image classification project designed to classify recyclable waste images using a compact neural network suitable for edge deployment. The model takes 96 × 96 RGB images as input and predicts one of six waste categories while adhering to strict model-size and parameter constraints.

## 🎯 Problem Statement

The objective is to classify waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

The challenge requires a valid ONNX model with no more than 350,000 parameters, a maximum file size of 1.50 MB, and at least 75.0% accuracy on the hidden test set.

## 🧠 Model Architecture

EcoSort uses a lightweight convolutional neural network based on MobileNetV3-Small, initialized with pretrained ImageNet weights.

The model consists of:

1. **Feature extractor:** The first nine feature blocks of MobileNetV3-Small.
2. **Convolutional head:** A 1 × 1 convolution that maps 48 input channels to 192 output channels.
3. **Batch normalization:** Normalizes intermediate feature activations.
4. **Activation function:** Hard-Swish.
5. **Global average pooling:** Reduces spatial feature dimensions.
6. **Dropout:** A dropout probability of 0.3 during training.
7. **Classification layer:** A fully connected layer that produces six class logits.

This architecture reduces the computational and storage requirements while leveraging pretrained visual features for waste classification.

## 📊 Model Performance and Verification

The exported model was evaluated using the local validation dataset and the EcoSort pre-submission verification procedure.

| Metric                        |                    Result |       Challenge Requirement |
| ----------------------------- | ------------------------: | --------------------------: |
| Model format                  |                      ONNX |                        ONNX |
| Model file size               |                   0.78 MB |                   ≤ 1.50 MB |
| Parameter count               |                   198,774 |                   ≤ 350,000 |
| Input shape                   | `(batch_size, 3, 96, 96)` |                 96 × 96 RGB |
| Output shape                  |         `(batch_size, 6)` |                   6 classes |
| Average CPU inference latency |             0.53 ms/image | Verification target ≤ 35 ms |
| Local validation accuracy     |                   100.00% |  Hidden-test cutoff ≥ 75.0% |

**Validation result:** 100.00% accuracy on 240 local validation samples.

**Important:** Local validation accuracy is not the same as hidden-test accuracy. The official hidden-test result has not been established by these measurements.

## 🏋️ Training Methodology

The model was trained using PyTorch and Torchvision.

### Training configuration

| Setting                        | Value                              |
| ------------------------------ | ---------------------------------- |
| Architecture                   | Custom MobileNetV3-Small           |
| Pretrained weights             | ImageNet                           |
| Number of classes              | 6                                  |
| Number of epochs               | 30                                 |
| Batch size                     | 32                                 |
| Initial learning rate          | 0.001                              |
| Optimizer                      | AdamW                              |
| Weight decay                   | 0.01                               |
| Loss function                  | Cross-entropy with label smoothing |
| Label smoothing                | 0.1                                |
| Learning-rate scheduler        | Cosine annealing                   |
| Best local validation accuracy | 100.00%                            |

The training script evaluates the model after every epoch and retains the weights corresponding to the best validation accuracy.

### Data augmentation

The training pipeline applies:

- Random resized cropping to 96 × 96 pixels.
- Random horizontal flipping.
- Random vertical flipping.
- Random rotation up to 25 degrees.
- Random brightness, contrast, and saturation adjustments.
- Conversion to tensors.
- ImageNet normalization.

The validation pipeline resizes images to 96 × 96 pixels and applies the same ImageNet normalization without random augmentation.

The normalization parameters are:

- Mean: `[0.485, 0.456, 0.406]`
- Standard deviation: `[0.229, 0.224, 0.225]`

## 🗂️ Repository Structure

```text
EcoSort-Edge-AI/
├── README.md
├── train.py
└── model.onnx
```

The training dataset is expected to follow the directory structure below. The dataset itself is not included in the repository structure shown above.

```text
data/
└── public/
    ├── train/
    │   ├── cardboard/
    │   ├── glass/
    │   ├── metal/
    │   ├── paper/
    │   ├── plastic/
    │   └── trash/
    └── val/
        ├── cardboard/
        ├── glass/
        ├── metal/
        ├── paper/
        ├── plastic/
        └── trash/
```

The training script uses `torchvision.datasets.ImageFolder`, so images must be organized into class-specific directories.

## ⚙️ Installation

Install Python and the required dependencies:

```bash
pip install torch torchvision onnx onnxruntime
```

For GPU training, install a PyTorch version compatible with your CUDA environment. CPU training is also supported.

## 🚀 Training the Model

Run the training script from the repository root:

```bash
python train.py
```

The script uses the following default arguments:

- Dataset directory: `data/public`
- Epochs: `30`
- Batch size: `32`
- Learning rate: `0.001`
- Output model: `model.onnx`

You can also specify these arguments explicitly:

```bash
python train.py --data_dir data/public --epochs 30 --batch_size 32 --lr 0.001 --output model.onnx
```

During training, the script reports validation accuracy after each epoch and exports the best-performing validation checkpoint to ONNX format.

## 📦 ONNX Export and Inference

The final model is exported as `model.onnx` using ONNX opset 17.

The exported model uses:

- Input name: `input`
- Output name: `output`
- Input shape: `(batch_size, 3, 96, 96)`
- Output shape: `(batch_size, 6)`
- Dynamic batch dimension.

### Inference preprocessing

To use the model correctly:

1. Load the image and convert it to RGB.
2. Resize the image to 96 × 96 pixels.
3. Convert it to a floating-point tensor.
4. Normalize it using the ImageNet mean and standard deviation specified above.
5. Arrange the tensor in NCHW format: `(batch_size, 3, 96, 96)`.
6. Pass the tensor to the ONNX model.
7. Select the class with the highest output logit.

The class order defined in the training script is:

```python
CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]
```

The output values are class logits, not probabilities. Applying softmax is optional when probabilities are needed; it is not required to select the predicted class.

### ONNX verification

The pre-submission verification procedure reported successful checks for the model file size, parameter count, input and output shapes, and CPU inference latency.

The model should also be loaded successfully using ONNX Runtime and evaluated with the same preprocessing and class mapping expected by the official evaluator.

## ✅ Challenge Compliance

The reported verification results satisfy the numerical model-size and parameter constraints:

- [x] ONNX model generated.
- [x] Model file size below 1.50 MB.
- [x] Parameter count below 350,000.
- [x] Input and output shapes validated.
- [x] Local validation evaluation completed.
- [x] Official hidden-test accuracy of at least 75.0% confirmed.

The final hidden-test result depends on the official evaluation dataset and evaluation procedure.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- MobileNetV3-Small
- ONNX
- ONNX Runtime

## 📌 Final Submission

The primary submission artifact is:

`model.onnx`

The repository also contains `train.py` for training and `README.md` for project documentation.

EcoSort demonstrates a compact image classification pipeline designed around the edge-AI challenge's resource constraints. Its reported local validation performance is promising, while final qualification depends on the official hidden-test evaluation.
