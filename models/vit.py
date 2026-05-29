import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits a CIFAR-10 image into non-overlapping patches and projects each patch
    into an embedding vector.

    For a 32x32 image and patch_size=4, the image is converted into an 8x8 grid,
    i.e. 64 patches. The Conv2d layer is equivalent to applying a learned linear
    projection to each flattened patch.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=80):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2

        # Kernel size and stride are both equal to patch_size, so patches do not overlap.
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        """
        Args:
            x: Input images of shape (batch_size, 3, 32, 32).

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embed_dim).
        """
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MLP(nn.Module):
    """
    Feed-forward network used inside each Transformer block.

    The hidden dimension is controlled by mlp_ratio. With embed_dim=80 and
    mlp_ratio=3, the hidden dimension is 240.
    """
    def __init__(self, embed_dim, mlp_ratio=3, dropout=0.1):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single pre-normalisation Transformer encoder block.

    Structure:
        LayerNorm -> Multi-Head Self-Attention -> Residual connection
        LayerNorm -> MLP -> Residual connection
    """
    def __init__(self, embed_dim=80, num_heads=4, mlp_ratio=3, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        """
        Args:
            x: Token sequence of shape (batch_size, num_tokens, embed_dim).

        Returns:
            Updated token sequence with the same shape.
        """
        # Self-attention sublayer with residual connection.
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + attn_output

        # Feed-forward sublayer with residual connection.
        x = x + self.mlp(self.norm2(x))

        return x


class ViT(nn.Module):
    """
    Compact Vision Transformer for CIFAR-10 classification.

    Architecture summary:
        - Input image size: 32x32
        - Patch size: 4x4
        - Number of patches: 64
        - Embedding dimension: 80
        - Transformer depth: 4 blocks
        - Attention heads: 4
        - MLP expansion ratio: 3
        - Output classes: 10

    The architecture is intentionally small and parameter-matched to the CNN
    baseline used in the project.
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=80,
        depth=4,
        num_heads=4,
        mlp_ratio=3,
        dropout=0.1,
    ):
        super().__init__()
        
        # Convert image into a sequence of patch embeddings.
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        # Learnable class token used as the final image representation.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings for the class token + all image patches.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)

        # Stack of Transformer encoder blocks.
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """
        Initialise learnable parameters.

        Truncated normal initialisation is commonly used in ViT-style models.
        Linear biases are initialised to zero.
        """
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Args:
            x: Batch of CIFAR-10 images with shape (batch_size, 3, 32, 32).

        Returns:
            Class logits with shape (batch_size, 10).
        """
        batch_size = x.shape[0]

        # Convert images into patch tokens.
        x = self.patch_embed(x)

        # Add one class token to each image sequence.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional information and apply dropout.
        x = x + self.pos_embed
        x = self.dropout(x)

        # Process the token sequence with Transformer blocks.
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use the final class-token representation for classification.
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits
        
def get_model():
    return ViT()
