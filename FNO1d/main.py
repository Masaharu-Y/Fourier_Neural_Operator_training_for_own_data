import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# NeuralOperator ライブラリ
from neuralop.models import FNO
from neuralop.training import Trainer
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses import LpLoss


@dataclass
class TrainConfig:
    """学習設定を管理するデータクラス"""
    seed: int = 42
    batch_size: int = 32
    resolution: int = 64
    n_modes: Tuple[int, int] = (16, 16)
    hidden_channels: int = 64
    n_epochs: int = 20
    learning_rate: float = 0.001
    save_dir: str = "./fig"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    """
    再現性のために乱数シードを固定します。
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
    辞書形式 {"x": ..., "y": ...} でデータを返す Dataset クラス。
    DefaultDataProcessor の仕様に合わせて使用します。
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
    
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx]}


def create_synthetic_data_2d(n_samples: int, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    2次元の学習用ダミーデータセットを作成します。
    
    Args:
        n_samples (int): サンプル数
        resolution (int): グリッド解像度

    Returns:
        inputs (torch.Tensor): 形状 (n_samples, 1, resolution, resolution)
        outputs (torch.Tensor): 形状 (n_samples, 1, resolution, resolution)
    """
    # 0から2πまでの2次元グリッドを作成
    x = torch.linspace(0, 2 * 3.14159, resolution)
    y = torch.linspace(0, 2 * 3.14159, resolution)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    input_list = []
    output_list = []
    
    for _ in range(n_samples):
        # ランダムな係数で2次元の波形を作成
        c1 = torch.randn(1)
        c2 = torch.randn(1)
        
        inp = c1 * torch.sin(grid_x) * torch.cos(grid_y) + \
              c2 * torch.cos(grid_x) * torch.sin(grid_y)
        
        # 出力関数（非線形変換）
        out = inp ** 2 + torch.sin(inp)
        
        # チャンネル次元を追加: (Res, Res) -> (1, Res, Res)
        input_list.append(inp.unsqueeze(0))
        output_list.append(out.unsqueeze(0))
    
    inputs = torch.stack(input_list)
    outputs = torch.stack(output_list)
    
    return inputs, outputs


def plot_result_2d(input_data: torch.Tensor, truth: torch.Tensor, prediction: torch.Tensor, 
                   title: str, save_path: str) -> None:
    """
    2次元データの入力、正解、予測をヒートマップで比較プロットして保存します。
    """
    # Tensor -> Numpy変換
    inp = input_data.squeeze().cpu().numpy()
    t = truth.squeeze().cpu().numpy()
    p = prediction.squeeze().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # プロット設定用リスト
    data_list = [inp, t, p]
    titles = ["Input", "Ground Truth", "Prediction"]
    
    for ax, data, t_str in zip(axes, data_list, titles):
        im = ax.imshow(data, cmap='viridis', origin='lower')
        ax.set_title(t_str)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def run_training(config: TrainConfig):
    """
    設定に基づいてデータの生成、モデルの構築、学習、評価を実行します。
    """
    print(f"Using device: {config.device}")
    os.makedirs(config.save_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 1. データの生成とロード
    # ---------------------------------------------------------
    print("Generating 2D data...")
    x_train, y_train = create_synthetic_data_2d(n_samples=500, resolution=config.resolution)
    x_test, y_test = create_synthetic_data_2d(n_samples=100, resolution=config.resolution)
    
    train_dataset = DictDataset(x_train, y_train)
    test_dataset = DictDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    # ---------------------------------------------------------
    # 2. 前処理・正規化
    # ---------------------------------------------------------
    # dim=[0, 2, 3]: Batch(0), Height(2), Width(3) を縮約
    in_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
    in_normalizer.fit(x_train)
    
    out_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
    out_normalizer.fit(y_train)
    
    data_processor = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer
    )
    data_processor = data_processor.to(config.device)
    
    # ---------------------------------------------------------
    # 3. モデル定義
    # ---------------------------------------------------------
    model = FNO(
        n_modes=config.n_modes,
        hidden_channels=config.hidden_channels,
        in_channels=1,
        out_channels=1
    )
    model = model.to(config.device)
    
    # ---------------------------------------------------------
    # 4. 学習前の評価 (Pre-training Check)
    # ---------------------------------------------------------
    print("Plotting pre-training result...")
    model.eval()
    with torch.no_grad():
        test_sample_x = x_test[0].unsqueeze(0).to(config.device)
        test_sample_y = y_test[0].unsqueeze(0).to(config.device)
        
        norm_x = in_normalizer(test_sample_x)
        out = model(norm_x)
        pred_pre = out_normalizer.inverse_transform(out)
        
        plot_result_2d(
            test_sample_x, test_sample_y, pred_pre, 
            "Pre-training Result (2D)", 
            os.path.join(config.save_dir, "pre_train_result.png")
        )

    # ---------------------------------------------------------
    # 5. 学習設定
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='sum')
    
    # ---------------------------------------------------------
    # 6. 学習実行
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
    # 7. 学習後の評価
    # ---------------------------------------------------------
    print("Plotting post-training result...")
    model.eval()
    with torch.no_grad():
        norm_x = in_normalizer(test_sample_x)
        out = model(norm_x)
        pred_post = out_normalizer.inverse_transform(out)
        
        loss = loss_fn(pred_post, test_sample_y)
        print(f"Test Sample Loss: {loss.item():.6f}")
        
        plot_result_2d(
            test_sample_x, test_sample_y, pred_post, 
            "Post-training Result (2D)", 
            os.path.join(config.save_dir, "post_train_result.png")
        )


def main():
    # 設定の初期化（ここでハイパーパラメータを変更可能）
    config = TrainConfig(
        seed=42,
        batch_size=32,
        n_epochs=20
    )
    
    # シード固定
    set_seed(config.seed)
    
    # 学習プロセスの実行
    run_training(config)


if __name__ == "__main__":
    main()