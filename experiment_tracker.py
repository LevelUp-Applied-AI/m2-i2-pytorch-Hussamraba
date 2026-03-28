import json
import time
import itertools

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class HousingModel(nn.Module):
    """
    Neural network for predicting housing prices from property features.

    Architecture: Linear(5, hidden_size) -> ReLU -> Linear(hidden_size, 1)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer1 = nn.Linear(5, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


def load_and_prepare_data():
    """Load housing data, standardize features, convert to tensors, and split 80/20."""
    df = pd.read_csv("data/housing.csv")

    feature_cols = ["area_sqm", "bedrooms", "floor", "age_years", "distance_to_center_km"]
    target_col = "price_jod"

    X = df[feature_cols]
    y = df[[target_col]]

    # Standardize features
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std

    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Fixed split for reproducibility
    torch.manual_seed(42)
    indices = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]

    split_idx = int(0.8 * len(X_tensor))
    X_train = X_shuffled[:split_idx]
    X_test = X_shuffled[split_idx:]
    y_train = y_shuffled[:split_idx]
    y_test = y_shuffled[split_idx:]

    return X_train, X_test, y_train, y_test


def compute_metrics(actual, predicted):
    """Compute MAE and R²."""
    actual_np = actual.numpy().flatten()
    pred_np = predicted.numpy().flatten()

    mae = np.mean(np.abs(actual_np - pred_np))

    ss_res = np.sum((actual_np - pred_np) ** 2)
    ss_tot = np.sum((actual_np - np.mean(actual_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return float(mae), float(r2)


def run_experiment(config, X_train, X_test, y_train, y_test):
    """Train one model configuration and return experiment results."""
    torch.manual_seed(42)

    model = HousingModel(hidden_size=config["hidden_size"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    start_time = time.time()

    for _ in range(config["num_epochs"]):
        model.train()
        train_preds = model(X_train)
        train_loss = criterion(train_preds, y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        final_train_preds = model(X_train)
        final_test_preds = model(X_test)

        final_train_loss = criterion(final_train_preds, y_train).item()
        final_test_loss = criterion(final_test_preds, y_test).item()

    test_mae, test_r2 = compute_metrics(y_test, final_test_preds)

    return {
        "config": {
            "learning_rate": config["learning_rate"],
            "hidden_size": config["hidden_size"],
            "num_epochs": config["num_epochs"],
        },
        "final_train_loss": float(final_train_loss),
        "final_test_loss": float(final_test_loss),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "training_time_sec": round(float(training_time), 4),
    }


def save_results(results, filename="experiments.json"):
    """Save experiment results to JSON."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {filename}")


def print_leaderboard(results, top_n=10):
    """Print top experiment results sorted by test MAE."""
    sorted_results = sorted(results, key=lambda r: r["test_mae"])

    print("\nTop Experiments by Test MAE")
    print("-" * 78)
    print(f"{'Rank':<5} {'LR':<8} {'Hidden':<8} {'Epochs':<8} {'Test MAE':<14} {'Test R²':<10} {'Time(s)':<10}")
    print("-" * 78)

    for rank, result in enumerate(sorted_results[:top_n], start=1):
        cfg = result["config"]
        print(
            f"{rank:<5} "
            f"{cfg['learning_rate']:<8} "
            f"{cfg['hidden_size']:<8} "
            f"{cfg['num_epochs']:<8} "
            f"{result['test_mae']:<14.2f} "
            f"{result['test_r2']:<10.4f} "
            f"{result['training_time_sec']:<10.4f}"
        )


def save_plot(results, filename="experiment_summary.png"):
    """Save a summary chart: test MAE vs learning rate, separated by hidden size."""
    plt.figure(figsize=(10, 6))

    hidden_sizes = sorted(set(r["config"]["hidden_size"] for r in results))

    for hidden_size in hidden_sizes:
        subset = [r for r in results if r["config"]["hidden_size"] == hidden_size]
        subset = sorted(subset, key=lambda r: (r["config"]["learning_rate"], r["config"]["num_epochs"]))

        x = [str(r["config"]["learning_rate"]) for r in subset]
        y = [r["test_mae"] for r in subset]

        plt.plot(x, y, marker="o", label=f"Hidden={hidden_size}")

    plt.title("Experiment Summary: Test MAE vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")


def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    learning_rates = [0.001, 0.005, 0.01, 0.05]
    hidden_sizes = [16, 32, 64]
    num_epochs_list = [50, 100, 200]

    configs = list(itertools.product(learning_rates, hidden_sizes, num_epochs_list))

    print(f"Running {len(configs)} experiments...\n")

    results = []

    for i, (lr, hidden, epochs) in enumerate(configs, start=1):
        config = {
            "learning_rate": lr,
            "hidden_size": hidden,
            "num_epochs": epochs,
        }

        print(
            f"[{i}/{len(configs)}] "
            f"lr={lr}, hidden_size={hidden}, num_epochs={epochs}"
        )

        result = run_experiment(config, X_train, X_test, y_train, y_test)
        results.append(result)

    save_results(results)
    print_leaderboard(results)
    save_plot(results)

    best_result = min(results, key=lambda r: r["test_mae"])
    print("\nBest Configuration:")
    print(best_result)


if __name__ == "__main__":
    main()