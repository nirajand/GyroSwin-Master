import torch
import numpy as np
import time
from typing import Tuple

class RealTimePlasmaInference:
    """
    High-performance inference engine for real-time gyro-kinetic analysis.
    Designed to run independently of the training environment.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load the compiled TorchScript model
        logger_info = f"Loading optimized model on {self.device}..."
        print(logger_info)
        
        # We use torch.jit.load which is faster and safer for production
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()
        
        # Warm up the GPU (prevents the first-run latency spike)
        self._warmup()

    def _warmup(self):
        """Standard production practice: run dummy data to initialize CUDA kernels."""
        with torch.no_grad():
            dummy = torch.randn(1, 4, 16, 16, 8, 8, device=self.device)
            for _ in range(5):
                _ = self.model(dummy)
        print("Warmup complete. Ready for inference.")

    @torch.no_grad()
    def predict(self, f_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs inference on a physical distribution function.
        
        Args:
            f_input: Numpy array of shape (T, X, Y, V, Mu)
        Returns:
            Tuple of (predicted_f_next, predicted_phi)
        """
        # 1. Pre-process: Convert to tensor and add batch dimension
        start_time = time.perf_counter()
        
        x = torch.from_numpy(f_input).float().unsqueeze(0).to(self.device)
        
        # 2. Run optimized inference
        # Using autocast for FP16 speedup on RTX 2060
        with torch.amp.autocast('cuda'):
            out_5d, out_3d = self.model(x)
        
        # 3. Post-process: Back to CPU/Numpy
        res_5d = out_5d.squeeze(0).cpu().numpy()
        res_3d = out_3d.squeeze(0).cpu().numpy()
        
        latency = (time.perf_counter() - start_time) * 1000
        print(f"Inference Latency: {latency:.2f} ms")
        
        return res_5d, res_3d

if __name__ == "__main__":
    # Path to the exported model from export.py
    DEPLOYED_MODEL = "deploy/gyronet_script.pt"
    
    try:
        engine = RealTimePlasmaInference(DEPLOYED_MODEL)
        
        # Simulate receiving a live data buffer from a simulation or sensor
        live_data = np.random.randn(4, 16, 16, 8, 8).astype(np.float32)
        
        # Predict
        f_next, phi = engine.predict(live_data)
        
        print(f"Predicted Potential Shape: {phi.shape}")
        
    except Exception as e:
        print(f"Deployment Error: {e}")
