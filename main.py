import torch
import numpy as np
from train import train_model
import config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    for i, seed in enumerate(config.RANDOM_SEEDS, 1):
        print(f"Run {i}/{config.NUM_RUNS}")
        results = train_model(config.DATA_ROOT, seed, device)
        all_results.append(results)
        print(f"ACC: {results['ACC']:.2f}%, AUC: {results['AUC']:.2f}%")

    print("\nFinal Results:")
    for metric in ['ACC', 'AUC']:
        values = [r[metric] for r in all_results]
        mean_val = np.mean(values)
        print(f"{metric}: {mean_val:.2f}%")


if __name__ == "__main__":
    main()
