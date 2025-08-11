import torch
import torch.nn as nn   

class PatchEmbeddings(nn.Module):
    """Class to convert a 2D image into 1D learnable embedding tensor"""
    def __init__(self,
                 img_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 192):
        super(PatchEmbeddings, self).__init__()
        
        assert img_size % patch_size == 0, f"Image size should be a multiple of patch size. Image size {img_size} and patch size {patch_size}"
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv2d patching layer
        self.proj = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
    
        # Flatten layer
        # Flatten the height and width dimension into a single dimension
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, embedding_dim, H', W')
        x = self.flatten(x) # (B, embedding_dim, H', W') -> (B, embedding_dim, num_patches)
        x = x.permute(0, 2, 1) # (B, embedding_dim, num_patches) -> (B, num_patches, embedding_dim)
        return x
    

class MultiheadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module -- MODIFIED to match timm's architecture.
    This version uses a single Linear layer for Q, K, V for efficiency and
    to allow direct weight loading from timm.
    """
    def __init__(self, embedding_dim=192, num_heads=3, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # MODIFICATION: A single Linear layer for Q, K, and V
        # The output dimension is 3 * embedding_dim because it holds Q, K, and V concatenated.
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=True)

        # The projection layer remains the same
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_patches, dim = x.shape

        # MODIFICATION: Project x once to get q, k, and v together
        # (B, N, D) -> (B, N, 3*D)
        qkv = self.qkv(x)

        # Reshape and split qkv into q, k, and v for multi-head attention
        # (B, N, 3*D) -> (B, N, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, n_patches, 3, self.num_heads, self.head_dim)
        
        # Permute to (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Split into q, k, v. Each will have shape (B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled Dot-Product Attention
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_score.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_patches, dim)
        
        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    
class MLP(nn.Module):
    """
    MLP block. Refactored to use named 'fc1' and 'fc2' like timm.
    """
    def __init__(self, embedding_dim=192, mlp_size:int=768, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, mlp_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_size, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class TransformerEncoderBlock(nn.Module):
    """Creates a ViT transformer encoder block"""
    def __init__(self,
                 embedding_dim:int=192,
                 num_heads:int=3,
                 mlp_size:int=768,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embedding_dim)

        # Create the MSA block
        self.attn = MultiheadSelfAttention(embedding_dim=embedding_dim,
                                               num_heads=num_heads,
                                               dropout=attn_dropout)
        
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Create teh MLP block
        self.mlp = MLP(embedding_dim=embedding_dim,
                            mlp_size=mlp_size,
                            dropout=mlp_dropout)

    def forward(self, X: torch.Tensor):
        # Create residual connection for MSA block
        X = self.attn(self.norm1(X)) + X

        # Create residual connection for MLP block
        X = self.mlp(self.norm2(X)) + X
        return X