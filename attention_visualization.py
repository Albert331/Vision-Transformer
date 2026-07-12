import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from dataclasses import dataclass
from transformers import SiglipVisionModel
import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def preprocess_image(image, image_size=224):
  preprocess = transforms.Compose([
      transforms.Resize((image_size,image_size)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485,0.456,0.406],
          std=[0.229,0.224,0.225]
      )
  ])
  image_tensor = preprocess(image)
  image_tensor = image_tensor.unsqueeze(0)
  return image_tensor



@dataclass
class SiglipVisionConfig:
  num_channels: int =3
  image_size: int = 224
  patch_size: int = 16
  num_attention_heads: int = 12
  hidden_size: int = 768
  attention_dropout:float = 0.0
  intermediate_size: int = 3072
  layer_norm_eps:float = 1e-6
  num_hidden_layers: int = 12


class SiglipVisionEmbeddings(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config

    self.num_channels = config.num_channels
    self.embed_dim = config.hidden_size
    self.image_size = config.image_size
    self.patch_size = config.patch_size

    self.patch_embedding = nn.Conv2d(
        in_channels=self.num_channels,
        out_channels=self.embed_dim,
        kernel_size=self.patch_size,
        stride=self.patch_size,
        padding='valid'
    )
    self.num_patches = (self.image_size//self.patch_size)**2
    self.num_positions = self.num_patches
    self.position_embedding = nn.Embedding(self.num_positions,self.embed_dim)
    self.register_buffer(
        'position_ids',
        torch.arange(self.num_positions).expand((1,-1)),
        persistent=False,
    )

  def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
    B, C, H, W = pixel_values.shape
    patch_embeds = self.patch_embedding(pixel_values)
    embeddings = patch_embeds.flatten(start_dim=2,end_dim=-1) 
    embeddings = embeddings.transpose(1,2)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings

class SiglipMLP(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
    self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = self.fc1(hidden_states)
    hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
    hidden_states = self.fc2(hidden_states)
    return hidden_states

class SiglipAttention(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.embed_dim = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.dropout = config.attention_dropout

    self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
    self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

  def forward(self, hidden_states,return_attn = False):
    B, T, C = hidden_states.shape

    q_states = self.q_proj(hidden_states)
    k_states = self.k_proj(hidden_states)
    v_states = self.v_proj(hidden_states)

    q_states = q_states.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    k_states = k_states.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
    v_states = v_states.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

    attn_weights = (q_states @ k_states.transpose(-2, -1)) * (1.0 / math.sqrt(k_states.size(-1)))
    attn_weights = F.softmax(attn_weights, dim=-1).to(q_states.dtype)
    attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_outs = attn_weights @ v_states
    attn_outs = attn_outs.transpose(1, 2)
    attn_outs = attn_outs.reshape(B, T, C).contiguous()
    attn_outs = self.out_proj(attn_outs)

    if return_attn:
      return attn_outs, attn_weights
    return attn_outs


class SiglipEncoderLayer(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.embed_dim = config.hidden_size
    self.self_attn = SiglipAttention(config)
    self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    self.mlp = SiglipMLP(config)
    self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

  def forward(self, hidden_states,return_attn = False):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    if return_attn:
        hidden_states, attn_weights = self.self_attn(hidden_states, return_attn=True)
    else:
        hidden_states = self.self_attn(hidden_states)    
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    
    if return_attn:
      return hidden_states, attn_weights
    return hidden_states

class SiglipEncoder(nn.Module):
  
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

  def forward(self, hidden_states):
    for layer in self.layers:
      hidden_states = layer(hidden_states)
    return hidden_states


class SiglipVisionTransformer(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.config = config
    self.embeddings = SiglipVisionEmbeddings(config)
    self.encoder = SiglipEncoder(config)
    self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

  def forward(self, pixel_values):
    hidden_states = self.embeddings(pixel_values)
    last_hidden_state = self.encoder(hidden_states)
    last_hidden_state = self.post_layernorm(last_hidden_state)
    return last_hidden_state


siglip = SiglipVisionTransformer(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))



hf_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
hf_model.eval()

siglip = SiglipVisionTransformer(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
siglip.eval()


class SiglipForClassification(nn.Module):
  def __init__(self,vision_transformer,hidden_size, num_classes):
    super().__init__()
    self.vision_transformer = vision_transformer
    self.classifier = nn.Linear(hidden_size,num_classes)

  def forward(self,pixel_values):
    last_hidden_state = self.vision_transformer(pixel_values)  
    pooled = last_hidden_state.mean(dim=1)
    logits = self.classifier(pooled)
    return logits
  


model_ft = SiglipForClassification(siglip, hidden_size=768, num_classes=10)

for param in model_ft.vision_transformer.parameters():
    param.requires_grad = False

for param in model_ft.classifier.parameters():
    param.requires_grad = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load('model_ft_trained.pt'))
model_ft = model_ft.to(device)
model_ft.eval()

ds = load_dataset("tanganke/eurosat")
item =ds['train'][0]
image = item['image'].convert('RGB')
image_tensor = preprocess_image(image).to(device)

def get_attention_map(siglip_model, image_tensor, layer_idx=-1):
    hidden_states = siglip_model.embeddings(image_tensor)

    for i, layer in enumerate(siglip_model.encoder.layers):
        if i == layer_idx or (layer_idx == -1 and i == len(siglip_model.encoder.layers) - 1):
            hidden_states, attn_weights = layer(hidden_states, return_attn=True)
        else:
            hidden_states = layer(hidden_states)

    return attn_weights   

with torch.no_grad():
    attn_weights = get_attention_map(siglip, image_tensor, layer_idx=2)


def visualize_attention(attn_weights, original_image, grid_size=14, patch_size=16, image_size=224):
    attn = attn_weights[0].mean(dim=0)   # [196, 196]

    attn_received = attn.mean(dim=0)     # [196]

    attn_grid = attn_received.reshape(grid_size, grid_size).cpu().numpy()

    attn_grid_resized = np.array(Image.fromarray(attn_grid).resize((image_size, image_size), Image.BILINEAR))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_image.resize((image_size, image_size)))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(original_image.resize((image_size, image_size)))
    axes[1].imshow(attn_grid_resized, cmap='jet', alpha=0.5)
    axes[1].set_title("Attention Overlay (last layer)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

siglip_pretrained = SiglipVisionTransformer(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
siglip_pretrained.eval()

hf_sd = hf_model.state_dict()
pretrained_sd = siglip_pretrained.state_dict()
for k in pretrained_sd:
    if k in hf_sd:
        pretrained_sd[k].copy_(hf_sd[k])
siglip_pretrained.load_state_dict(pretrained_sd)
siglip_pretrained = siglip_pretrained.to(device)

with torch.no_grad():
    attn_pretrained = get_attention_map(siglip_pretrained, image_tensor, layer_idx=-1)
    attn_finetuned = get_attention_map(siglip, image_tensor, layer_idx=-1)  
row, col = 7, 9   
print(f"Pixel region: rows {row*16}-{row*16+16}, cols {col*16}-{col*16+16}")


resized_img = image.resize((224, 224))
crop = resized_img.crop((col*16, row*16, col*16+16, row*16+16))
plt.imshow(crop)
plt.title(f"Patch ({row},{col}) - most affected by fine-tuning")
plt.show()