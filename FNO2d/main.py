import os
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# NeuralOperator library
from neuralop.models import FNO
from neuralop.training import Trainer
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses import LpLoss


class DictDataset(Dataset):
    """
    Dataset that returns data in dictionary format {"x": ..., "y": ...}.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx]}


def create_synthetic_data_2d(n_samples: int = 1000, resolution: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a synthetic 2D dataset for training.

    Input: a superposition of random 2D waves on a grid
    Output: a simple nonlinear mapping: input squared plus sine(input)

    Returns:
        inputs (torch.Tensor): shape (n_samples, 1, resolution, resolution)
        outputs (torch.Tensor): shape (n_samples, 1, resolution, resolution)
    """
    # Create a 2D grid from 0 to 2π
    x = torch.linspace(0, 2 * 3.14159, resolution)
    y = torch.linspace(0, 2 * 3.14159, resolution)
    # Use indexing='ij' to match matrix coordinate system
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    input_list = []
    output_list = []
    
    for _ in range(n_samples):
        # Create a 2D waveform with random coefficients
        # e.g.: c1 * sin(X)cos(Y) + c2 * cos(X)sin(Y)
        c1 = torch.randn(1)
        c2 = torch.randn(1)
        
        inp = c1 * torch.sin(grid_x) * torch.cos(grid_y) + \
              c2 * torch.cos(grid_x) * torch.sin(grid_y)
        
        # Output function (nonlinear transformation)
        out = inp ** 2 + torch.sin(inp)
        
        # Add channel dimension: (Res, Res) -> (1, Res, Res)
        input_list.append(inp.unsqueeze(0))
        output_list.append(out.unsqueeze(0))
    
    # リストを結合してバッチテンソルへ: (Batch, Channels, Height, Width)
    inputs = torch.stack(input_list)
    outputs = torch.stack(output_list)
    
    return inputs, outputs


def plot_result_2d(input_data: torch.Tensor, truth: torch.Tensor, prediction: torch.Tensor, title: str, save_path: str) -> None:
    """
    Plot and save heatmaps comparing input, ground truth and prediction for 2D data.
    """
    # Tensor -> NumPy conversion (remove Batch and Channel dims -> Height, Width)
    inp = input_data.squeeze().cpu().numpy()
    t = truth.squeeze().cpu().numpy()
    p = prediction.squeeze().cpu().numpy()
    
    # Create 3 subplots (Input, Ground Truth, Prediction)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 入力データ
    im0 = axes[0].imshow(inp, cmap='viridis', origin='lower')
    axes[0].set_title(f"Input")
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. 正解データ
    im1 = axes[1].imshow(t, cmap='viridis', origin='lower')
    axes[1].set_title(f"Ground Truth")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. 予測データ
    im2 = axes[2].imshow(p, cmap='viridis', origin='lower')
    axes[2].set_title(f"Prediction")
    axes[2].axis('off')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 32
    RESOLUTION = 64
    # For 2D FNO specify tuple (height_modes, width_modes)
    N_MODES = (16, 16)
    HIDDEN_CHANNELS = 64
    N_EPOCHS = 20  # 2D is more computationally expensive; use fewer epochs (increase if needed)
    LR = 0.001
    
    # Prepare save directory
    os.makedirs("./fig_2d", exist_ok=True)

    # ---------------------------------------------------------
    # 2. Data generation and loading (Data Preparation)
    # ---------------------------------------------------------
    print("Generating 2D data...")
    x_train, y_train = create_synthetic_data_2d(n_samples=500, resolution=RESOLUTION)
    x_test, y_test = create_synthetic_data_2d(n_samples=100, resolution=RESOLUTION)
    
    train_dataset = DictDataset(x_train, y_train)
    test_dataset = DictDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ---------------------------------------------------------
    # 3. Preprocessing & Normalization
    # ---------------------------------------------------------
    # For 2D data shape is (Batch, Channel, Height, Width).
    # Using dim=[0, 2, 3] reduces over Batch(0), Height(2), Width(3)
    # to compute statistics per Channel(1).
    in_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
    in_normalizer.fit(x_train)
    
    out_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
    out_normalizer.fit(y_train)
    
    data_processor = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer
    )
    data_processor = data_processor.to(device)
    
    # ---------------------------------------------------------
    # 4. Model Definition
    # ---------------------------------------------------------
    # Passing a tuple (16, 16) to n_modes initializes an FNO2d model
    model = FNO(
        n_modes=N_MODES,
        hidden_channels=HIDDEN_CHANNELS,
        in_channels=1,
        out_channels=1
    )
    model = model.to(device)
    
    # ---------------------------------------------------------
    # 5. Pre-training Evaluation
    # ---------------------------------------------------------
    print("Plotting pre-training result...")
    model.eval()
    with torch.no_grad():
        test_sample_x = x_test[0].unsqueeze(0).to(device)
        test_sample_y = y_test[0].unsqueeze(0).to(device)
        
        norm_x = in_normalizer(test_sample_x)
        out = model(norm_x)
        pred_pre = out_normalizer.inverse_transform(out)
        
        # Also plot the input (test_sample_x)
        plot_result_2d(test_sample_x, test_sample_y, pred_pre, 
                      "Pre-training Result (2D)", 
                      "./fig_2d/pre_train_result.png")

    # ---------------------------------------------------------
    # 6. Training setup (Optimizer, Scheduler, Loss)
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    
    loss_fn = LpLoss(d=2, p=2, reduction='sum')
    
    # ---------------------------------------------------------
    # 7. Training (using Trainer)
    # ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        n_epochs=N_EPOCHS,
        device=device,
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
    # 8. Post-training Evaluation
    # ---------------------------------------------------------
    print("Plotting post-training result...")
    model.eval()
    with torch.no_grad():
        norm_x = in_normalizer(test_sample_x)
        out = model(norm_x)
        pred_post = out_normalizer.inverse_transform(out)
        
        loss = loss_fn(pred_post, test_sample_y)
        print(f"Test Sample Loss: {loss.item():.6f}")
        
        # Also plot the input (test_sample_x)
        plot_result_2d(test_sample_x, test_sample_y, pred_post, 
                      "Post-training Result (2D)", 
                      "./fig_2d/post_train_result.png")

if __name__ == "__main__":
    main()