import torch
import torch.nn as nn
from src.model import AdvancedGyroNet
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Export")

def export_model(checkpoint_path: str, export_dir: str = "deploy"):
    """
    Converts a Lightning Checkpoint to TorchScript and ONNX.
    """
    export_path = Path(export_dir)
    export_path.mkdir(exist_ok=True)

    # 1. Load Model from Checkpoint
    logger.info(f"Loading weights from {checkpoint_path}")
    model = AdvancedGyroNet.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to("cpu") # Exporting on CPU is more portable

    # 2. Create Dummy Input (Match the 5D distribution shape)
    # Shape: (Batch, Time, X, Y, V, Mu)
    dummy_input = torch.randn(1, 4, 16, 16, 8, 8)

    # 3. Export to TorchScript (JIT)
    # Using 'trace' is preferred for Transformers with fixed sequence lengths
    logger.info("Exporting to TorchScript...")
    script_path = export_path / "gyronet_script.pt"
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(script_path))
        logger.info(f"TorchScript saved to {script_path}")
    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")

    # 4. Export to ONNX
    # This format is best for TensorRT / NVIDIA Triton Inference Server
    logger.info("Exporting to ONNX...")
    onnx_path = export_path / "gyronet.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,  # Supports complex Transformer ops
        do_constant_folding=True,
        input_names=['input_f'],
        output_names=['output_5d', 'output_3d'],
        dynamic_axes={
            'input_f': {0: 'batch_size'}, # Allow variable batch sizes
            'output_5d': {0: 'batch_size'},
            'output_3d': {0: 'batch_size'}
        }
    )
    logger.info(f"ONNX saved to {onnx_path}")

    # 5. Numerical Validation
    logger.info("Validating exported model...")
    with torch.no_grad():
        orig_out_5d, orig_out_3d = model(dummy_input)
        
        # Load and run script
        loaded_script = torch.jit.load(str(script_path))
        script_5d, script_3d = loaded_script(dummy_input)
        
        # Check parity
        diff = torch.abs(orig_out_5d - script_5d).max()
        logger.info(f"Max numerical deviation: {diff:.2e}")
        
        if diff < 1e-5:
            logger.info("Export Validation PASSED")
        else:
            logger.warning("Numerical deviation detected!")

if __name__ == "__main__":
    # Point this to your actual checkpoint file in the logs/ directory
    latest_ckpt = "logs/lightning_logs/version_0/checkpoints/last.ckpt"
    if Path(latest_ckpt).exists():
        export_model(latest_ckpt)
    else:
        logger.error("No checkpoint found. Please train the model first.")
