import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from dataclasses import dataclass
from transformers import SiglipVisionModel
import math

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


# Re-defining SiglipVisionConfig from MnUR-YBRvprK to ensure it's available and consistent
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

# Re-defining SiglipVisionEmbeddings from MnUR-YBRvprK with the fix
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
    embeddings = patch_embeds.flatten(start_dim=2,end_dim=-1) # FIX APPLIED HERE
    embeddings = embeddings.transpose(1,2)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings

# Re-defining SiglipMLP from zxh3gXe9uvK8
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

# Re-defining SiglipAttention from I6gGFLDcoMOO
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

  def forward(self, hidden_states):
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

    return attn_outs

# Re-defining SiglipEncoderLayer from -cgh7YkLx7pf
class SiglipEncoderLayer(nn.Module):
  def __init__(self, config: SiglipVisionConfig):
    super().__init__()
    self.embed_dim = config.hidden_size
    self.self_attn = SiglipAttention(config)
    self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    self.mlp = SiglipMLP(config)
    self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

  def forward(self, hidden_states):
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states = self.self_attn(hidden_states)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states

class SiglipEncoder(nn.Module):
  """The conveyor belt: holds 12 stations, pushes data through all of them in order."""
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


# load pretrained HF model
hf_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
hf_model.eval()

siglip = SiglipVisionTransformer(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
siglip.eval()

# copy all matching weights
hf_sd = hf_model.state_dict()
our_sd = siglip.state_dict()

missing = []
for k in our_sd:
    if k in hf_sd:
        our_sd[k].copy_(hf_sd[k])
    else:
        missing.append(k)

siglip.load_state_dict(our_sd)
print("Unmatched keys:", missing)   # should print an empty list
ds = load_dataset("tanganke/eurosat")
print(ds)

class EuroSATDataset(Dataset):
  def __init__(self, hf_dataset):
    self.hf_dataset = hf_dataset

  def __len__(self):
    return len(self.hf_dataset)

  def __getitem__(self, idx):
    item = self.hf_dataset[idx]
    image = item['image'].convert('RGB')
    pixel_values = preprocess_image(image).squeeze(0)  
    label = item['label']
    return pixel_values,label

train_dataset = EuroSATDataset(ds['train'].shuffle(seed=42).select(range(2000))) 
test_dataset = EuroSATDataset(ds['test'].shuffle(seed=42).select(range(500)))   

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

optimizer = torch.optim.AdamW(model_ft.classifier.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epoch = 5

for epoch in range(num_epoch):
  model_ft.train()
  total_loss = 0

  for pixel_values, labels in train_loader:
    pixel_values, labels = pixel_values.to(device), labels.to(device)

    logits = model_ft(pixel_values)
    loss = criterion(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  avg_loss = total_loss / len(train_loader)                        # ← now indented, runs every epoch
  print(f"Epoch {epoch+1}/{num_epoch}, Avg Loss: {avg_loss:.4f}")   # ← now indented, runs every epoch

