import torch
from torch import Tensor
import torch.fft
from typing import Tuple, Optional, Union

class KineticFieldSolver:
    """
    A production-grade solver for kinetic moments and the Poisson equation.
    Uses spectral methods to solve the Laplacian with singularity handling.
    """
    def __init__(
        self, 
        grid_size: Tuple[int, int, int], 
        domain_length: Tuple[float, float, float],
        device: Optional[torch.device] = None,
        eps_0: float = 1.0
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nx, self.ny, self.nz = grid_size
        self.lx, self.ly, self.lz = domain_length
        self.eps_0 = eps_0
        
        # Pre-compute k-space kernels to avoid redundant calculations in the main loop
        self._init_spectral_kernel()

    def _init_spectral_kernel(self) -> None:
        """Pre-calculates the 1/|k|^2 kernel."""
        # Calculate wave numbers: k = 2 * pi * n / L
        kx = torch.fft.fftfreq(self.nx, d=self.lx/self.nx).to(self.device) * 2 * torch.pi
        ky = torch.fft.fftfreq(self.ny, d=self.ly/self.ny).to(self.device) * 2 * torch.pi
        kz = torch.fft.fftfreq(self.nz, d=self.lz/self.nz).to(self.device) * 2 * torch.pi
        
        # Create 3D meshgrid for k^2 = kx^2 + ky^2 + kz^2
        KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        k_sq = KX**2 + KY**2 + KZ**2
        
        # Calculate 1/k^2 kernel. Handle the k=0 (DC) singularity.
        # Physics: The mean potential is arbitrary; we set the DC component to 0.
        inv_k_sq = 1.0 / k_sq
        inv_k_sq[0, 0, 0] = 0.0 
        
        # Registration as a buffer ensures it moves with the model to GPU/CPU
        self.register_buffer("inv_k_sq", inv_k_sq)

    def register_buffer(self, name: str, tensor: Tensor):
        """Helper to manage internal state tensors."""
        setattr(self, name, tensor.to(self.device))

    @torch.jit.script_method
    def get_velocity_moments(
        self, 
        f: Tensor, 
        v_par: Tensor, 
        dv: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Computes 0th (density) and 1st (flux) moments.
        Shape f: [Batch, X, Y, Z, V_par, V_mu]
        """
        # Ensure velocity broadcast [1, 1, 1, 1, V_par, 1]
        v_view = v_par.view(1, 1, 1, 1, -1, 1)
        
        # Density (n) = Integral(f) dv
        density = torch.sum(f, dim=(-1, -2)) * dv
        
        # Parallel Flux (Gamma) = Integral(v_par * f) dv
        flux = torch.sum(v_view * f, dim=(-1, -2)) * dv
        
        return density, flux

    def solve_poisson(self, density: Tensor) -> Tensor:
        """
        Solves -grad^2(phi) = (density - mean_density) / eps_0
        
        Args:
            density: 4D Tensor [Batch, X, Y, Z]
        Returns:
            Electrostatic potential phi [Batch, X, Y, Z]
        """
        # 1. Enforce quasi-neutrality (remove DC component)
        # sum(rho) must be zero for periodic Poisson solvers
        rho = (density - torch.mean(density, dim=(1, 2, 3), keepdim=True)) / self.eps_0
        
        # 2. Forward FFT to Fourier space
        rho_k = torch.fft.fftn(rho, dim=(1, 2, 3))
        
        # 3. Solve in Fourier space: phi_k = rho_k / |k|^2
        # self.inv_k_sq is [X, Y, Z], rho_k is [Batch, X, Y, Z]
        phi_k = rho_k * self.inv_k_sq
        
        # 4. Inverse FFT to real space
        phi = torch.fft.ifftn(phi_k, dim=(1, 2, 3)).real
        
        return phi
