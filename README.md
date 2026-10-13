# Image Classification Using MobileNetV2

## About the Project
This project performs image classification using a pretrained MobileNetV2 model with PyTorch and Torchvision. It predicts the most likely categories for an input image and displays the top 5 predictions with their confidence scores.

## Features
- Classifies images using a pretrained deep learning model.
- Displays the top 5 predicted categories.
- Shows confidence scores for each prediction.
- Supports JPG, JPEG, and PNG images.
- Downloads pretrained model weights automatically on the first run.

## Technologies Used
- Python
- PyTorch
- Torchvision
- Pillow (PIL)

## Installation

Install the required dependencies using:

```bash
python -m pip install -r requirements.txt
```

## Usage

Place your image in the project folder and run:

```bash
python image_classifier.py --image animal.jpeg
```

Replace `animal.jpeg` with the name of your image file.

To display a different number of predictions, use the `--top` argument:

```bash
python image_classifier.py --image animal.jpeg --top 3
```

## Example Output

```text
Predictions for: animal.jpeg
1. lesser panda: 43.29%
2. giant panda: 0.53%
3. polecat: 0.45%
4. red fox: 0.44%
5. kit fox: 0.40%
```

*Note: Predictions and confidence scores vary depending on the input image. The model uses ImageNet categories and may not recognize specific monuments or objects by their exact names.*

## Project Structure

```text
Image-Classification/
├── image_classifier.py
├── requirements.txt
├── README.md
├── animal.jpeg
└── ele.jpeg
```

## Model
MobileNetV2 pretrained on ImageNet.

## License
This project is available for educational purposes. Add a specific open-source license if you intend to permit reuse under defined terms.