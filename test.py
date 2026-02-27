from PIL import Image
import torch
import clip

print("Imported clip from:", clip.__file__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
text = clip.tokenize([
    "a diagram",
    "a dog",
    "a cat",
    "text logo",
    "a screenshot",
]).to(device)

with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu()[0]

print("Probabilities:")
for label, prob in zip(["a diagram", "a dog", "a cat", "text logo", "a screenshot"], probs):
    print(f"{label}: {prob.item():.4f}")