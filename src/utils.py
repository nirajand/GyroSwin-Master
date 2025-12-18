import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class KineticSolver5D(nn.Module):
    """
    5D Plasma Solver: 3D Real Space (X, Y, Z) + 2D Velocity Space (V_par, V_mu).
    Inherits from nn.Module for seamless GPU/CPU transfers and optimization.
    """
    def __init__(
        self, 
        grid_size: Tuple[int, int, int], 
        domain_length: Tuple[float, float, float],
        eps_0: float = 1.0
    ):
        super().__init__()
        self.nx, self.ny, self.nz = grid_size
        self.lx, self.ly, self.lz = domain_length
        self.eps_0 = eps_0
        
        # Pre-compute and register the 3D spectral kernel
        self.register_buffer("inv_k_sq", self._init_spectral_kernel())

    def _init_spectral_kernel(self) -> Tensor:
        # Calculate wave numbers for 3D
        kx = torch.fft.fftfreq(self.nx, d=self.lx/self.nx) * 2 * torch.pi
        ky = torch.fft.fftfreq(self.ny, d=self.ly/self.ny) * 2 * torch.pi
        kz = torch.fft.fftfreq(self.nz, d=self.lz/self.nz) * 2 * torch.pi
        
        # 3D meshgrid
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_sq = KX**2 + KY**2 + KZ**2
        
        # Numerical safety: avoid division by zero at the DC component
        inv_k_sq = 1.0 / torch.where(k_sq == 0, torch.ones_like(k_sq), k_sq)
        inv_k_sq[0, 0, 0] = 0.0  # Physical constraint: zero mean potential
        return inv_k_sq

    def get_moments(self, f: Tensor, v_par: Tensor, dv: float) -> Tuple[Tensor, Tensor]:
        """
        Reduces 5D distribution function to 3D spatial moments.
        Input f: (B, NX, NY, NZ, V_par, V_mu)
        """
        # Align v_par for broadcasting across the spatial and V_mu dims
        # View: (1, 1, 1, 1, V_par, 1)
        v_view = v_par.view(1, 1, 1, 1, -1, 1)
        
        # Integrate over the last two dimensions (Velocity Space)
        density = torch.sum(f, dim=(-1, -2)) * dv
        flux = torch.sum(v_view * f, dim=(-1, -2)) * dv
        
        return density, flux

    def solve_poisson(self, density: Tensor) -> Tensor:
        """
        Solves -grad^2(phi) = (n - <n>) / eps_0 in 3D.
        Input density: (B, NX, NY, NZ)
        """
        # 1. Background subtraction (Quasi-neutrality)
        rho = (density - torch.mean(density, dim=(1, 2, 3), keepdim=True)) / self.eps_0
        
        # 2. 3D FFT
        rho_k = torch.fft.fftn(rho, dim=(1, 2, 3))
        
        # 3. Poisson solve in Fourier space
        phi_k = rho_k * self.inv_k_sq
        
        # 4. Inverse 3D FFT
        phi = torch.fft.ifftn(phi_k, dim=(1, 2, 3)).real
        return phi
