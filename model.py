from utils import *

class Attention(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        # Define Q, K, V matrices here
        self.Q = nn.Linear(hidden_dim, hidden_dim)
        self.K = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        B, Seq_len, _ = x.shape
        
        # Project inputs
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)
        
        # Split into heads
        Q_new = rearrange(query, 'b s (n h) -> b n s h', n=self.n_heads)
        K_new = rearrange(key, 'b s (n h) -> b n s h', n=self.n_heads)
        V_new = rearrange(value, 'b s (n h) -> b n s h', n=self.n_heads)
        
        # Compute attention scores
        attn = Q_new @ rearrange(K_new, 'b n s h -> b n h s')  # shape: (b, n, s, s)
        attn = attn / np.sqrt(self.head_dim)                   # scale by sqrt(d_k)
        
        # Softmax
        score = F.softmax(attn, dim=-1)                        # (b, n, s, s)
        
        # Weighted sum
        output = score @ V_new                                 # (b, n, s, h)
        
        # Merge heads
        out = rearrange(output, 'b n s h -> b s (n h)')
        
        return out

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.input_dim = input_dim
        
        self.down_proj = nn.Linear(input_dim,  64)
        self.up_proj = nn.Linear(64,  input_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.down_proj(x)
        out = self.relu(out)
        out = self.up_proj(out)
        
        return out
    
    
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Conv2d extracts patches & projects them
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x shape: (batch, channels, H, W)
        x = self.proj(x) # (batch, embed_dim, H/patch, W/patch)
        x = x.flatten(2) # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2) # (batch, n_patches, embed_dim)
        return x
        
    
class Model(nn.Module):
    def __init__(
        self,
        img_size = 32, 
        patch_size = 4, 
        hidden_dim = 64, 
        in_channels = 3, 
        n_heads = 4,
        depth = 4
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.depth = depth
        
        self.patch_embed = PatchEmbedding(img_size = self.img_size, 
                                          patch_size = self.patch_size, 
                                          in_channels = self.in_channels,
                                          embed_dim = self.hidden_dim,
                                          )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches+1, self.hidden_dim))
        
        self.attention = nn.ModuleList([Attention(self.hidden_dim, self.n_heads) for _ in range(self.depth)])
        
        self.mlp = MLP(self.hidden_dim)
        
        self.down_proj = nn.Linear(self.hidden_dim, 10)
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.patch_embed(x)     # (B, N, D)
        B, N, D = out.shape
        
        # Expand cls_token for the whole batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        
        # Prepend cls token to sequence
        out = torch.cat((cls_tokens, out), dim=1)      # (B, N+1, D)
        
        # Add positional embedding
        out = out + self.pos_embed[:, :N+1, :]
        
        # Transformer blocks
        for blk in self.attention:
            out = blk(out)
            out = self.relu(out)
            out = self.mlp(out)
            out = self.relu(out)
            out = self.layer_norm(out)
        
        # Extract CLS token (index 0)
        cls_out = out[:, 0]   # (B, D)
        
        # Classification head
        out = self.down_proj(cls_out)
        
        return out

    
        
        
        