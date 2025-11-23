import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# NeuralOperator Library
from neuralop.models import FNO
from neuralop.training import Trainer
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses import LpLoss


@dataclass
class TrainConfig:
    """Data class to manage training configurations."""
    seed: int = 42
    batch_size: int = 32
    resolution: int = 64
    # Number of modes for 1D FNO (specified as a tuple)
    n_modes: Tuple[int] = (16,)
    hidden_channels: int = 64
    n_epochs: int = 50
    learning_rate: float = 0.001
    save_dir: str = "./fig"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


class DictDataset(Dataset):
    """
    Dataset class that returns data in dictionary format {"x": ..., "y": ...}.
    Used to match the specification of DefaultDataProcessor.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx]}


def create_synthetic_data_1d(n_samples: int, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a synthetic 1D dataset for training.
    
    Args:
        n_samples (int): Number of samples
        resolution (int): Grid resolution

    Returns:
        inputs (torch.Tensor): Shape (n_samples, 1, resolution)
        outputs (torch.Tensor): Shape (n_samples, 1, resolution)
    """
    # Grid from 0 to 2pi
    x = torch.linspace(0, 2 * 3.14159, resolution)
    
    input_list = []
    output_list = []
    
    for _ in range(n_samples):
        # Create waveform with random coefficients: c1*sin(x) + c2*cos(2x)
        c1 = torch.randn(1)
        c2 = torch.randn(1)
        inp = c1 * torch.sin(x) + c2 * torch.cos(2 * x)
        
        # Output function (nonlinear transformation): inp^2 + sin(inp)
        out = inp ** 2 + torch.sin(inp)
        
        # Add channel dimension: (Res) -> (1, Res)
        input_list.append(inp.unsqueeze(0))
        output_list.append(out.unsqueeze(0))
    
    inputs = torch.stack(input_list)
    outputs = torch.stack(output_list)
    
    return inputs, outputs


def plot_result_1d(input_data: torch.Tensor, truth: torch.Tensor, prediction: torch.Tensor, 
                   title: str, save_path: str) -> None:
    """
    Plots and saves input, ground truth, and prediction for 1D data.
    Left: Input Data
    Right: Ground Truth vs Prediction
    """
    # Tensor -> Numpy conversion
    inp = input_data.squeeze().cpu().numpy()
    t = truth.squeeze().cpu().numpy()
    p = prediction.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Plot Input Data
    axes[0].plot(inp, label="Input", color='blue', alpha=0.7)
    axes[0].set_title("Input Data")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Plot Comparison of Ground Truth and Prediction
    axes[1].plot(t, label="Ground Truth", color='black', linewidth=2)
    axes[1].plot(p, label="Prediction", color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Ground Truth vs Prediction")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def run_training(config: TrainConfig):
    """
    Executes data generation, model construction, training, and evaluation based on the configuration.
    """
    print(f"Using device: {config.device}")
    os.makedirs(config.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Data Generation and Loading
    # ---------------------------------------------------------
    print("Generating 1D data...")
    x_train, y_train = create_synthetic_data_1d(n_samples=1000, resolution=config.resolution)
    x_test, y_test = create_synthetic_data_1d(n_samples=200, resolution=config.resolution)
    
    train_dataset = DictDataset(x_train, y_train)
    test_dataset = DictDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # ---------------------------------------------------------
    # 2. Preprocessing & Normalization
    # ---------------------------------------------------------
    # 1D data (Batch, Channel, Resolution)
    # dim=[0, 2] reduces Batch(0) and Resolution(2), computing statistics per Channel(1)
    in_normalizer = UnitGaussianNormalizer(dim=[0, 2])
    in_normalizer.fit(x_train)
    
    out_normalizer = UnitGaussianNormalizer(dim=[0, 2])
    out_normalizer.fit(y_train)
    
    data_processor = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer
    )
    data_processor = data_processor.to(config.device)
    
    # ---------------------------------------------------------
    # 3. Model Definition
    # ---------------------------------------------------------
    # Initialized as FNO1d (since n_modes=(16,))
    model = FNO(
        n_modes=config.n_modes,
        hidden_channels=config.hidden_channels,
        in_channels=1,
        out_channels=1
    )
    model = model.to(config.device)
    
    # ---------------------------------------------------------
    # 4. Pre-training Evaluation
    # ---------------------------------------------------------
    print("Plotting pre-training result...")
    model.eval()
    with torch.no_grad():
        test_sample_x = x_test[0].unsqueeze(0).to(config.device)
        test_sample_y = y_test[0].unsqueeze(0).to(config.device)
        
        norm_x = in_normalizer(test_sample_x)
        out = model(norm_x)
        pred_pre = out_normalizer.inverse_transform(out)
        
        plot_result_1d(
            test_sample_x, test_sample_y, pred_pre, 
            "Pre-training Result (1D)", 
            os.path.join(config.save_dir, "pre_train_result.png")
        )

    # ---------------------------------------------------------
    # 5. Training Setup
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='sum')
    
    # ---------------------------------------------------------
    # 6. Execute Training
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        n_epochs=config.n_epochs,
        device=config.device,
        data_processor=data_processor,
        wandb_log=False,
        verbose=True
    )
    
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        test_loaders={"default": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=False,
        training_loss=loss_fn,
        eval_losses={"l2": loss_fn}
    )
    print("Training finished!")

    # ---------------------------------------------------------
    # 7. Post-training Evaluation
    # ---------------------------------------------------------
    print("Plotting post-training result...")
    model.eval()
    with torch.no_grad():
        norm_x = in_normalizer(test_sample_x)
        out = model(norm_x)
        pred_post = out_normalizer.inverse_transform(out)
        
        loss = loss_fn(pred_post, test_sample_y)
        print(f"Test Sample Loss: {loss.item():.6f}")
        
        plot_result_1d(
            test_sample_x, test_sample_y, pred_post, 
            "Post-training Result (1D)", 
            os.path.join(config.save_dir, "post_train_result.png")
        )


def main():
    # Initialize Configuration
    config = TrainConfig(
        seed=42,
        batch_size=32,
        n_epochs=50,
        n_modes=(16,)  # Mode setting for 1D
    )
    
    # Set seed
    set_seed(config.seed)
    
    # Run training process
    run_training(config)


if __name__ == "__main__":
    main()