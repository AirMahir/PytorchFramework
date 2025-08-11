import torch
import torch.nn as nn
from models.modules import PatchEmbeddings, TransformerEncoderBlock

class ViT(nn.Module):
    """Creates a vision transformer model"""
    def __init__(self, config):
        img_size =  config["img_size"]                
        in_channels = 3             
        patch_size = config["patch_size"]                 
        num_transformer_layers = 12
        embedding_dim = config["embed_dims"]    
        mlp_size = 768           
        num_heads = 3              
        attn_dropout = 0    
        mlp_dropout = 0.1          
        embedding_dropout = 0.1 # Dropout for patch and position embeddings
        num_classes = config["num_classes"]
        
        super(ViT, self).__init__()
        
        assert img_size % patch_size == 0, f"Image size should be a multiple of patch size. Image size {img_size} and patch size {patch_size}"
        
        # Creating a patch embeddings layer
        self.patch_embed = PatchEmbeddings(img_size=img_size,
                                                 in_channels=in_channels,
                                                 patch_size=patch_size,
                                                 embedding_dim=embedding_dim)
        
        # Creating a dropout for patch embeddings
        self.pos_drop = nn.Dropout(p=embedding_dropout)
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))


        # Creating a transformer encoder blocks
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(embedding_dim=embedding_dim,
                                        num_heads=num_heads,
                                        mlp_size=mlp_size,
                                        mlp_dropout=mlp_dropout,
                                        attn_dropout=attn_dropout)
                for _ in range(num_transformer_layers)
            ]
        )

        self.norm = nn.LayerNorm(normalized_shape=embedding_dim)
        # Creating the classifier head
        self.head = nn.Linear(in_features=embedding_dim, out_features=num_classes)

    def forward(self, X: torch.Tensor):
        batch_size = X.shape[0]
        X = self.patch_embed(X)
        # Prepend class token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        X = torch.cat((cls_token, X), dim=1)
        
        # Add positional embedding
        X = X + self.pos_embed
        X = self.pos_drop(X)
        X = self.blocks(X)
        # Passing the 0 indexed logits through classifier
        X = self.norm(X)
        X = self.head(X[:, 0])
        return X