from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(device)
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
model.eval()


ds = load_dataset("tanganke/eurosat")
class_names = ds['train'].features['label'].names


text_prompts = [f"a satellite photo of {name}" for name in class_names]


correct = 0
total = 0

for i in range(200):   # sample of 200 test images
    example = ds['test'][i]
    image = example['image'].convert('RGB')
    true_idx = example['label']

    inputs = processor(text=text_prompts, images=image, return_tensors="pt", padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits_per_image)

    pred_idx = probs.argmax(dim=1).item()
    correct += (pred_idx == true_idx)
    total += 1

print(f"Zero-shot accuracy: {100 * correct / total:.2f}%")