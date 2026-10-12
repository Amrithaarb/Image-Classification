"""
Image classification using a pretrained MobileNetV2 model.
Run: python image_classifier.py --image path/to/image.jpg
The first run downloads pretrained weights and ImageNet class labels.
"""
import argparse
from pathlib import Path

try:
    import torch
    import torchvision.models
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Run: python -m pip install -r requirements.txt"
    ) from exc

def classify_image(image_path, top_k=5):
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.eval()
    preprocess = weights.transforms()

    with Image.open(path) as image:
        image = image.convert("RGB")
        batch = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.nn.functional.softmax(model(batch)[0], dim=0)

    scores, indices = torch.topk(probabilities, k=min(top_k, len(probabilities)))
    categories = weights.meta["categories"]
    return [(categories[index], float(score)) for score, index in zip(scores, indices)]

def main():
    parser = argparse.ArgumentParser(description="Classify an image using pretrained MobileNetV2.")
    parser.add_argument("--image", required=True, help="Path to a JPG or PNG image")
    parser.add_argument("--top", type=int, default=5, help="Number of predictions to show")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")

    try:
        predictions = classify_image(args.image, args.top)
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"\nPredictions for: {args.image}")
    for rank, (label, probability) in enumerate(predictions, start=1):
        print(f"{rank}. {label}: {probability * 100:.2f}%")

if __name__ == "__main__":
    main()