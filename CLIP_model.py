from transformers import AutoTokenizer, CLIPModel
import torch
import pickle as pkl

with open("processed_dataset/train.pkl", "rb") as f:
    train_dataset = pkl.load(f)

with open("processed_dataset/val.pkl", "rb") as f:
    val_dataset = pkl.load(f)

with open("processed_dataset/test.pkl", "rb") as f:
    test_dataset = pkl.load(f)


from transformers import AutoTokenizer, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# tokenizer 























# processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# texts = ["a cat"]
# image = train_dataset[:1]["image"]
# inputs = processor(text=texts, images=image * len(texts), padding=True, return_tensors="pt")

# outputs = model(**inputs)

# logits = outputs.logits
# print(logits.shape)
# print(logits)

# import torch
# import numpy as np
# import matplotlib.pyplot as plt

# # Convert logits to a numpy array
# seg_maps = logits.detach().cpu().numpy()  # Shape: (3, 352, 352)

# # Normalize each segmentation map separately
# for i, seg_map in enumerate(seg_maps):
#     seg_map = (seg_map - seg_map.min()) / (seg_map.max() - seg_map.min())  # Normalize to [0, 1]
#     seg_map = (seg_map * 255).astype(np.uint8)  # Scale to [0, 255]

#     # Display the segmentation map
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1,2,1)
#     plt.imshow(seg_map, cmap="gray")
#     plt.axis("off")
#     plt.title("prediction")

#     plt.subplot(1,2,2)
#     plt.imshow(train_dataset[:1]["mask"][0].resize((224,224)), cmap="gray")
#     plt.axis("off")
#     plt.title("target")

# plt.show()
