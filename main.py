import torch
from PIL import Image
import clip
from pathlib import Path
import adversarial_data as noise
import cv2

# .venv\Scripts\activate
# path = Path('image_testing')

# 1. Load the model and processor
print("Imported clip from:", clip.__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Main
# labels = ["cat", "dog", "forest", "office", "desk", "library", "library desk", "parrot", "monkey", "snake"]
# labels2 = ["parrot", "office", "desk", "library", "sky", "glasses", "desert"] # Book
labels2 = ["church outdoor", "shop", "monestary", "chapel", "library", "building", "road", "street"]
# Tokenize and encode
text_inputs = clip.tokenize([f"a photo of a {c}" for c in labels2]).to(device)

predictions = {}
names = []
n = 5
path = Path('image_testing')


for entry in path.iterdir():
    # Adversarial data
    # image_gauss_variants = noise.iterate_gaussian_noise(f"{path}/{entry.name}", n)
    # image_salt_variants = noise.iterate_salt_pepper(f"{path}/{entry.name}", n)
    # rotated_variants = noise.rotation(f"{path}/{entry.name}")
    # brightened_variants = noise.brightness(f"{path}/{entry.name}", 4, 50)
    pixelated_variants = noise.pixelate(f"{path}/{entry.name}", n)

    names.append(entry.name)
    predictions[entry.name] = []

    # Predicting
    plan = {4: 25} # Prune to 25 tokens after Layer 4
    for image in pixelated_variants:
        # 1. Convert BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Convert NumPy array to PIL Image
        pil_img = Image.fromarray(img_rgb)

        # 3. Apply CLIP preprocessing
        # This returns a Torch Tensor: shape [3, 224, 224]
        image = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu()[0]
        for label, prob in zip(labels2, probs):
            prob = float(prob)
        predictions[entry.name].append(dict(zip(labels2, probs)))
        import pdb; pdb.set_trace()


# Printing the results
counter = 0
i = 0
# For each image
for item in predictions:
    # For each variant of this image
    for i, image_score in enumerate(predictions[item]):
        print(f"Filename: {item}. {i + 1}/{n}")
        print()
        for label, score in image_score.items():
            print(f"{label}: {score:.2%}")
        print("")
    print("")
    print("")

    counter += 1
    if counter % n == 0:
        counter = 0
        i += 1