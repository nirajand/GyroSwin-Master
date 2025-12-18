import os
import sys
import logging
import torch
import lightning as L
from pathlib import Path

# Import our refactored production modules
from src.engine import PlasmaDataEngine, SimulationConfig
from src.model import AdvancedGyroNet
from src.pipeline import GyroDataModule

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_system_check():
    """Verify hardware and directory structure before starting."""
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

def main():
    run_system_check()

    # 1. Physical Simulation / Data Generation Phase
    # In production, we check if data exists to avoid redundant generation
    zarr_path = "data/plasma_master.zarr"
    if not os.path.exists(zarr_path):
        logger.info("Zarr dataset not found. Starting Data Engine...")
        sim_cfg = SimulationConfig(
            samples=500,        # Higher sample count for better generalization
            chunk_size=50,       # Vectorized batch writing
            spatial_res=(16, 16),
            v_res=(8, 8)
        )
        engine = PlasmaDataEngine(sim_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
        engine.generate(path=zarr_path)
    else:
        logger.info(f"Existing dataset found at {zarr_path}. Skipping generation.")

    # 2. Pipeline Initialization
    # We use a batch size suited for the 6GB VRAM constraint (RTX 2060)
    dm = GyroDataModule(
        path=zarr_path, 
        batch_size=16, 
        num_workers=4,
        train_val_split=0.8
    )

    # 3. Model Architecture Configuration
    model = AdvancedGyroNet(
        spatial_dim=(16, 16),
        velocity_dim=(8, 8),
        embed_dim=128,
        num_blocks=4
    )

    # 4. Lightning Trainer Configuration
    # Optimized for speed and low-memory hardware
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=20,
        precision="16-mixed",         # 16-bit precision for Tensor Core acceleration
        log_every_n_steps=5,
        default_root_dir="logs/",
        enable_progress_bar=True,
        deterministic=True            # Ensures reproducibility for physics research
    )

    # 5. Execution
    logger.info("Starting Training Pipeline...")
    try:
        trainer.fit(model, datamodule=dm)
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
