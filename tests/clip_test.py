import torch
import clip
from PIL import Image

model, preprocess = clip.load("ViT-B/32", device="cuda")

print("done")

labels = {}

for line in open("C:\\Users\\armaa\\PycharmProjects\\Skribbl.io\\_tokenization.txt").read().splitlines():
    labels[line] = "Pixel drawing of " + line

labels_subset = {key: value for key, value in labels.items() if len(key.strip()) == 10}

image = preprocess(Image.open("sunglasses.png").convert("RGB")).unsqueeze(0).cuda()
text = clip.tokenize([label for label in labels_subset.values()]).cuda()

with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = sorted({key: value for key, value in zip(labels_subset.keys(), list(logits_per_image.softmax(dim=-1).cpu().numpy())[0])}.items(), key=lambda x: x[1], reverse=True)

    print(dict(probs[:4]))
