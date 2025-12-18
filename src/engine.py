import os
import zarr
import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

# Assuming the KineticFieldSolver from previous refactor is in src.physics
from src.physics import KineticFieldSolver 

@dataclass
class SimulationConfig:
    samples: int = 100
    time_steps: int = 4
    spatial_res: Tuple[int, int] = (16, 16)
    v_res: Tuple[int, int] = (8, 8)
    domain_size: Tuple[float, float] = (1.0, 1.0)
    chunk_size: int = 10  # Write in batches to optimize Zarr I/O

class PlasmaDataEngine:
    def __init__(self, config: SimulationConfig, device: str = "cpu"):
        self.cfg = config
        self.device = torch.device(device)
        
        # Initialize Physics Solver (Pre-computes K-space kernels)
        self.solver = KineticFieldSolver(
            grid_size=self.cfg.spatial_res,
            domain_length=self.cfg.domain_size,
            device=self.device
        )
        
        # Velocity grid [V_par]
        self.v_par = torch.linspace(-3, 3, self.cfg.v_res[0], device=self.device)
        self.dv = float(self.v_par[1] - self.v_par[0])

    def generate(self, path: str = "data/plasma_master.zarr"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Define Shapes
        shape_5d = (self.cfg.samples, self.cfg.time_steps, *self.cfg.spatial_res, *self.cfg.v_res)
        shape_3d = (self.cfg.samples, self.cfg.time_steps, *self.cfg.spatial_res)
        
        # Initialize Zarr with optimized chunking
        # Chunks should be large enough to saturate disk I/O but fit in RAM
        store = zarr.DirectoryStore(path)
        root = zarr.group(store=store, overwrite=True)
        
        f_ds = root.create_dataset("f", shape=shape_5d, chunks=(self.cfg.chunk_size, 1, *self.cfg.spatial_res, *self.cfg.v_res), dtype='f4')
        phi_ds = root.create_dataset("phi", shape=shape_3d, chunks=(self.cfg.chunk_size, 1, *self.cfg.spatial_res), dtype='f4')
        q_ds = root.create_dataset("q", shape=(self.cfg.samples, 1), dtype='f4')

        print(f"--- Generating {self.cfg.samples} Samples (Batch Size: {self.cfg.chunk_size}) ---")

        for i in range(0, self.cfg.samples, self.cfg.chunk_size):
            actual_batch = min(self.cfg.chunk_size, self.cfg.samples - i)
            
            # 1. Generate Batch on Device (Vectorized)
            # f_batch: [B, T, X, Y, V, Mu]
            f_batch = torch.randn(actual_batch, self.cfg.time_steps, *self.cfg.spatial_res, *self.cfg.v_res, device=self.device)
            
            # 2. Compute Physics Moments
            # Reshape T into Batch to process all time-steps in parallel
            f_reshaped = f_batch.view(-1, *self.cfg.spatial_res, *self.cfg.v_res)
            density, _ = self.solver.get_velocity_moments(f_reshaped, self.v_par, self.dv)
            
            # 3. Solve Poisson (Quasi-neutrality)
            phi_reshaped = self.solver.solve_poisson(density)
            phi_batch = phi_reshaped.view(actual_batch, self.cfg.time_steps, *self.cfg.spatial_res)
            
            # 4. Compute Target (Heat Flux / Variance)
            q_batch = torch.mean(f_batch**2, dim=(1, 2, 3, 4, 5)).view(-1, 1)

            # 5. Async-like write to Zarr
            f_ds[i : i + actual_batch] = f_batch.cpu().numpy()
            phi_ds[i : i + actual_batch] = phi_batch.cpu().numpy()
            q_ds[i : i + actual_batch] = q_batch.cpu().numpy()
            
            print(f"Progress: {i + actual_batch}/{self.cfg.samples}")

if __name__ == "__main__":
    config = SimulationConfig(samples=1000, chunk_size=20)
    engine = PlasmaDataEngine(config, device="cuda")
    engine.generate()
