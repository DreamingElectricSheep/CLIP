from matplotlib import text
import torch
from PIL import Image
import clip
from pathlib import Path
import adversarial_data as noise
import cv2
import numpy as np


# 1. Load the model and processor
print("Imported clip from:", clip.__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = 16
model, preprocess = clip.load(f"ViT-B/{model_name}", device=device)


def visualize_pruning(image_bgr, topk_indices, grid_size, dim_factor=0.3):
    """
    image_bgr: NumPy array of any size (H, W, 3).
    topk_indices: The indices (0-48) returned by the transformer.
    grid_size: The square root of the number of patches (usually 7 for ViT-B).
    dim_factor: Transparency of pruned areas.
    """
    if topk_indices is None:
        return image_bgr
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
        # cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 255, 255), 1)

    return vis_img


# Main
labels2 = ["church outdoor",  "coffee shop", "conference center", "squirrel monkey", "scuba diver", "red panda", "ladybird", "hamburger", "tree", "shop", "monestary", "library", "building", "road", "steak", "pizza", "fish", "eagle"]
# Tokenize and encode
text_inputs = clip.tokenize([f"a photo of a {c}" for c in labels2]).to(device)

predictions = {}
names = []
n = 5
path = Path('image_testing')


repetitions = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
# repetitions = ["a", "b", "c", "d", "e"]
box_size = 4
center_patch = (8, 6) # Center
# center_patch = None

# repetitions = ["a"]
if model_name == 32:
    pruning_plans = [None, {2: 25}, {6: 25}, {10: 25}] # For B/32
    grid_size = 7
elif model_name == 16:
    # pruning_plans = [None, {2: 98}]
    pruning_plans = [None, {2: 98}, {6: 98}, {10: 98}] # For B/16
    # pruning_plans = [{2: 78}, {2: 58}, {2: 38},{2: 18}] # For more constrained testing of dropping tokens from layer 2
    grid_size = 14

print(f"--- Starting experiment with model ViT-B/{model_name} ---")
for pruning_plan in pruning_plans:
    # For each repetition (to get multiple data points for each image)
    for rep in repetitions:
        # For every image
        for entry in path.iterdir():
            # Adversarial data
            image_gauss_variants = noise.iterate_gaussian_noise(f"{path}/{entry.name}", n, grid_size, box_size, center_patch)
            # image_salt_variants = noise.iterate_salt_pepper(f"{path}/{entry.name}", n)
            # rotated_variants = noise.rotation(f"{path}/{entry.name}")
            # brightened_variants = noise.brightness(f"{path}/{entry.name}", 4, 50)
            # pixelated_variants = noise.pixelate(f"{path}/{entry.name}", n)

            if entry.stem not in names:
                names.append(entry.stem)

            layer_key = list(pruning_plan.keys())[0] if pruning_plan else "None"
            num_kept = list(pruning_plan.values())[0] if pruning_plan else "All"

            predictions[f"{entry.name}_{layer_key}_{num_kept}_{rep}"] = []

            # Predicting
            for i, image in enumerate(image_gauss_variants):
                # 1. Keep a BGR copy for visualization (OpenCV style)
                copy = image.copy()
                
                # 2. Convert to RGB for the CLIP model
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)

                # 3. Apply CLIP preprocessing
                # This returns a Torch Tensor: shape [3, 224, 224]
                image = preprocess(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits_per_image, logits_per_text, all_indices = model(image, text_inputs, pruning_plan)
                    probs = logits_per_image.softmax(dim=-1).cpu()[0]
                    # Store results including the plan metadata for easier CSV exporting later
                result_entry = {
                    "variant": i,
                    "plan_label": f"Layer_{layer_key}",
                    "layer": layer_key,
                    "kept": num_kept,
                    "scores": dict(zip(labels2, probs.tolist()))
                }
                predictions[f"{entry.name}_{layer_key}_{num_kept}_{rep}"].append(result_entry)

                # --- Visualizing ---
                # Extract indices for the specific layer we just pruned
                top_indices = all_indices.get(layer_key, [None])[0] if pruning_plan else None
                result_view = visualize_pruning(copy, top_indices, grid_size = grid_size)
                
                # Saving the image
                # vis_filename = f"pruning_vis/{model_name}/constrained_tokens/{entry.stem}_p/{entry.stem}_gauss_{layer_key}_{num_kept}_{rep}_{i}.png"
                vis_filename = f"pruning_vis/{model_name}p/{entry.stem}/{entry.stem}_gauss_{layer_key}_{num_kept}_{rep}_{i}.png"
                cv2.imwrite(vis_filename, result_view)
    print(f"--- Completed pruning plan: {pruning_plan} ---")

# Exporting to CSV
import csv

# 1. Define the CSV filename
for name in names:
    # csv_file = f"experiment_data/{model_name}/constrained_tokens/{name}_p_experiment_data.csv"
    csv_file = f"experiment_data/{model_name}p/{name}_p_experiment_data.csv"

    header = ["Filename", "Pruning_Layer", "Tokens_Kept", "Variant_ID"] + labels2

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for filename, results in predictions.items():
            # Filtering results for each image
            if filename.split(".")[0] == name:
                for res in results:
                    row = [
                        filename, 
                        res["layer"], 
                        res["kept"],
                        res["variant"] 
                    ] + [res["scores"].get(lbl, 0) for lbl in labels2]
                    writer.writerow(row)
            else:
                continue

    print(f"--- Export Complete! Data saved to {csv_file} ---")
