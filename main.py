import torch
from PIL import Image
import clip
from pathlib import Path
import adversarial_data as noise
import cv2
import numpy as np

# .venv\Scripts\activate
# path = Path('image_testing')

# 1. Load the model and processor
print("Imported clip from:", clip.__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def visualize_pruning(image_bgr, topk_indices, grid_size=7, dim_factor=0.3):
    """
    image_bgr: NumPy array of any size (H, W, 3).
    topk_indices: The indices (0-48) returned by the transformer.
    grid_size: The square root of the number of patches (usually 7 for ViT-B).
    dim_factor: Transparency of pruned areas.
    """
    h, w, _ = image_bgr.shape
    
    # Dynamically calculate patch dimensions based on actual image size
    patch_h = h / grid_size
    patch_w = w / grid_size
    
    # Create the dimmed base image
    vis_img = (image_bgr.astype(float) * dim_factor).astype(np.uint8)
    
    if torch.is_tensor(topk_indices):
        indices = topk_indices.cpu().numpy().flatten()
    else:
        indices = topk_indices

    for idx in indices:
        # Map the flat index (0-48) to grid coordinates
        row = idx // grid_size
        col = idx % grid_size
        
        # Calculate pixel coordinates for this specific image size
        y1, y2 = int(row * patch_h), int((row + 1) * patch_h)
        x1, x2 = int(col * patch_w), int((col + 1) * patch_w)
        
        # Restore the original "kept" patch
        vis_img[y1:y2, x1:x2] = image_bgr[y1:y2, x1:x2]
        
        # Draw white border for clarity
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return vis_img


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

# For every image
for entry in path.iterdir():
    # Adversarial data
    image_gauss_variants = noise.iterate_gaussian_noise(f"{path}/{entry.name}", n)
    # image_salt_variants = noise.iterate_salt_pepper(f"{path}/{entry.name}", n)
    # rotated_variants = noise.rotation(f"{path}/{entry.name}")
    # brightened_variants = noise.brightness(f"{path}/{entry.name}", 4, 50)
    # pixelated_variants = noise.pixelate(f"{path}/{entry.name}", n)

    names.append(entry.name)
    predictions[entry.name] = []

    # Predicting
    plan = {4: 25} # Prune to 25 tokens after Layer 4
    for image in image_gauss_variants:
        # 1. Convert BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        copy = img_rgb.copy()
        
        # 2. Convert NumPy array to PIL Image
        pil_img = Image.fromarray(img_rgb)

        # 3. Apply CLIP preprocessing
        # This returns a Torch Tensor: shape [3, 224, 224]
        image = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text, all_indices = model(image, text_inputs)
            probs = logits_per_image.softmax(dim=-1).cpu()[0]
        for label, prob in zip(labels2, probs):
            prob = float(prob)
        predictions[entry.name].append(dict(zip(labels2, probs)))
        import pdb; pdb.set_trace()
        top_indices = all_indices[5][0] # Layer 6, Batch Index 0
        # 3. Generate and save the visualization
        result_view = visualize_pruning(copy, top_indices)
        cv2.imwrite("pruning_vis.png", result_view)


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