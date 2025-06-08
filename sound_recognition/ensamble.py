from pathlib import Path

import numpy as np


def pred_geometric_mean(preds_set):
    result = np.ones_like(preds_set[0])
    for preds in preds_set:
        result *= preds
    return result ** (1.0 / len(preds_set))


def ensemble(folder: Path, prefix: str):
    files = list(folder.glob(f"{prefix}_pred*.npy"))
    if not files:
        print(f"No prediction files found in {folder} for {prefix}.")
        return None
    preds_set = np.array([np.load(f) for f in files])
    result = pred_geometric_mean(preds_set)
    np.save(folder / f"ensemble_{prefix}_preds.npy", result)
    print(f"Saved ensemble to {folder / f'ensemble_{prefix}_preds.npy'}")
    return result


def main():
    for folder_name in ["X", "LH"]:
        folder = Path(folder_name)
        print(f"== Ensemble in [{folder}] ==")
        ensemble(folder, "train")
        ensemble(folder, "test")


if __name__ == "__main__":
    main()
