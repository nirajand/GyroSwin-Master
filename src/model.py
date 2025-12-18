import torch
import torch.nn as nn
import lightning as L
from xformers.ops import memory_efficient_attention, LowerTriangularMask
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Dict

class FusedAttentionBlock(nn.Module):
    """
    Optimized Transformer Block using xFormers for memory efficiency.
    Includes Pre-Norm architecture for better gradient flow.
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm Residual Path for Attention
        res = x
        x = self.norm1(x)
        B, N, C = x.shape
        
        # Reshape for xFormers: [B, N, Heads, Head_Dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, -1)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        # Memory Efficient Attention (FlashAttention-2 backend if available)
        attn = memory_efficient_attention(q, k, v)
        attn = attn.reshape(B, N, C)
        x = res + self.attn_dropout(attn)

        # Pre-Norm Residual Path for MLP
        x = x + self.mlp(self.norm2(x))
        return x

class AdvancedGyroNet(L.LightningModule):
    def __init__(
        self, 
        spatial_dim: Tuple[int, int] = (16, 16),
        velocity_dim: Tuple[int, int] = (8, 8),
        embed_dim: int = 128,
        num_blocks: int = 4,
        lr: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        self.spatial_res = spatial_dim
        self.v_res = velocity_dim
        
        # Patch Projection: Flattening the 2D velocity slices into tokens
        # Each 'token' represents a spatial (X,Y) coordinate containing (V, Mu) info
        self.input_proj = nn.Linear(velocity_dim[0] * velocity_dim[1], embed_dim)
        
        # Transformer Backbone
        self.blocks = nn.ModuleList([
            FusedAttentionBlock(embed_dim) for _ in range(num_blocks)
        ])
        
        # Multi-task Heads
        # 1. 5D Head: Predicts distribution function [B, N_tokens, V*Mu]
        self.head_5d = nn.Linear(embed_dim, velocity_dim[0] * velocity_dim[1])
        
        # 2. 3D Head: Predicts Potential Field [B, N_tokens, 1]
        self.head_3d = nn.Linear(embed_dim, 1)

    def forward(self, f_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        f_in: (Batch, Time, X, Y, V, Mu) -> Flattened to (Batch * Time * X * Y, V * Mu)
        """
        B, T, X, Y, V, M = f_in.shape
        # Flatten spatial/time into batch for the transformer to treat them as sequences
        # Here we treat (X*Y) as the sequence length N
        x = f_in.view(B * T, X * Y, V * M)
        
        x = self.input_proj(x)
        
        for block in self.blocks:
            if self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        # Project back to physical dimensions
        out_5d = self.head_5d(x).view(B, T, X, Y, V, M)
        out_3d = self.head_3d(x).view(B, T, X, Y)
        
        return out_5d, out_3d

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        f_in, f_target, phi_target = batch
        pred_5d, pred_3d = self(f_in)
        
        # Multi-task Loss Scaling (Production systems often use Uncertainty Weighting)
        loss_5d = nn.functional.mse_loss(pred_5d, f_target)
        loss_3d = nn.functional.mse_loss(pred_3d, phi_target)
        
        # Total Loss with importance weighting
        total_loss = loss_5d + 0.5 * loss_3d
        
        self.log_dict({
            "train/total_loss": total_loss,
            "train/loss_5d": loss_5d,
            "train/loss_3d": loss_3d
        }, prog_bar=True, sync_dist=True)
        
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
