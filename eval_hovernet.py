import argparse
import numpy as np
import pandas as pd

from pathlib import Path

from segmentation.utils.metrics import get_conic_metrics, get_multi_r2


def main():

    # Get arguments
    parser = argparse.ArgumentParser(description='Conic Challenge - Evaluation')
    parser.add_argument('--path', '-p', required=True, type=str, help='Path to Hover-Net results')
    args = parser.parse_args()

    # Paths
    path_results = Path(args.path)

    preds = np.load(path_results / 'valid_pred.npy')
    gts = np.load(path_results / 'valid_true.npy')

    print(f"Calculate metrics:")
    metrics_df = get_conic_metrics(gts, preds)
    metrics = np.squeeze(metrics_df.values)

    # r2 metric
    pred_counts = pd.read_csv(path_results / "valid_pred_cell.csv")
    gt_counts = pd.read_csv(path_results / "valid_true_cell.csv")

    r2 = get_multi_r2(gt_counts, pred_counts)
    print(f"  R2: {r2}")

    result = pd.DataFrame([[path_results.parent.stem, metrics[2], metrics[3], metrics[4], metrics[5], metrics[6],
                            metrics[7], metrics[0], metrics[1], r2]],
                          columns=["model_name", "multi_pq+ (neu)", "multi_pq+ (epi)", "multi_pq+ (lym)",
                                   "multi_pq+ (pla)", "multi_pq+ (eos)",  "multi_pq+ (con)", "multi_pq+",
                                   "pq_metrics_avg", "R2"])

    result.to_csv(Path(__file__).parent / "scores_isbi_hovernet.csv",
                  header=not (Path(__file__).parent / "scores_isbi_hovernet.csv").exists(),
                  index=False,
                  mode="a")


if __name__ == "__main__":
    main()
