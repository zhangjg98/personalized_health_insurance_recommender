# thresholds.py
import pandas as pd

def compute_dynamic_thresholds(csv_path, keys, lower_quantile=0.25, upper_quantile=0.75, scale_factor=1.3, use_z_score=False):
    """
    Compute dynamic thresholds (e.g., lower and upper percentiles or z-scores) for the given keys.
    
    Parameters:
      csv_path (str): Path to the processed user-item CSV file.
      keys (list): List of column names for which to compute thresholds.
      lower_quantile (float): Lower percentile (e.g., 0.25).
      upper_quantile (float): Upper percentile (e.g., 0.75).
      scale_factor (float): Factor to scale thresholds for more variation.
      use_z_score (bool): Whether to use z-scores for thresholding.
    
    Returns:
      dict: A dictionary mapping each key to a dictionary of lower and upper thresholds.
    """
    df = pd.read_csv(csv_path, index_col=0)
    print("CSV Columns:", df.columns)  # Debugging log
    thresholds = {}
    for key in keys:
        if key in df.columns:
            if use_z_score:
                mean = df[key].mean()
                std = df[key].std()
                lower = mean - (1.0 * std)  # 1 standard deviation below mean
                upper = mean + (1.0 * std)  # 1 standard deviation above mean
            else:
                lower = df[key].quantile(lower_quantile) * scale_factor
                upper = df[key].quantile(upper_quantile) * scale_factor
            thresholds[key] = {"low": lower, "high": upper}

            # Debugging logs
            print(f"Computed thresholds for {key}: low={lower}, high={upper}")
        else:
            print(f"Key {key} not found in CSV columns!")  # Debugging log
    return thresholds

def unified_thresholds(csv_path, keys):
    """
    Unified threshold calculation for all modules.
    """
    return compute_dynamic_thresholds(csv_path, keys, lower_quantile=0.1, upper_quantile=0.9, scale_factor=1.3)

if __name__ == "__main__":
    # Expanded list of variables for threshold calculations
    keys = [
        "TOT_MDCR_STDZD_PYMT_PC", 
        "TOT_MDCR_PYMT_PC",
        "BENE_AVG_RISK_SCRE", 
        "IP_CVRD_STAYS_PER_1000_BENES",
        "ER_VISITS_PER_1000_BENES",
        "MA_PRTCPTN_RATE"
    ]
    thresholds = compute_dynamic_thresholds("processed_user_item_matrix.csv", keys, scale_factor=1.3, use_z_score=True)
    print("Dynamic thresholds:", thresholds)
