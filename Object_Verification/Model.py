import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# CLIP modelini ve işlemciyi yükleme
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    return inputs

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=1)

def main():
    # Görüntüleri yükleme ve işleme
    image1 = load_and_preprocess_image("/Users/dilan/PycharmProjects/Object_Verification/Project/images/whitecup.jpg")
    image2 = load_and_preprocess_image("/Users/dilan/PycharmProjects/Object_Verification/Project/images/lamp.jpg")

    with torch.no_grad():
        embedding1 = model.get_image_features(**image1)
        embedding2 = model.get_image_features(**image2)

    # Embedding'leri normalize etme
    embedding1 /= embedding1.norm(dim=-1, keepdim=True)
    embedding2 /= embedding2.norm(dim=-1, keepdim=True)

    # Benzerlik hesaplama
    similarity = cosine_similarity(embedding1, embedding2)
    print(f"İki görüntü arasındaki benzerlik: {similarity.item()}")

    # Karar verme
    threshold = 0.7  # Eşik değeri
    if similarity > threshold:
        print("Bu objeler aynıdır.")
    else:
        print("Bu objeler farklıdır.")

if __name__ == "__main__":
    main()
