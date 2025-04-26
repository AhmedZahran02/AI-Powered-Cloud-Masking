
import pandas as pd
import numpy as np
import pandas.api.types
import sys
import pandas as pd


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def rle_decode(mask_rle, shape):
    """
    Decodes an RLE-encoded string into a binary mask.
    """
    if not isinstance(mask_rle, str):
        # If NaN or float or None, treat as empty mask
        return np.zeros(shape, dtype=np.uint8)

    mask_rle = mask_rle.strip()
    if not mask_rle:
        return np.zeros(shape, dtype=np.uint8)

    s = list(map(int, mask_rle.split()))
    starts, lengths = s[0::2], s[1::2]

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    return mask.reshape(shape, order='F')

def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Computes the Dice coefficient between two binary masks."""
    intersection = np.sum(mask1 * mask2)
    return (2.0 * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-7)  # Avoid division by zero

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """Computes the Dice score between solution and submission."""
    
    # Check if required columns exist
    required_columns = {row_id_column_name, "segmentation"}
    if not required_columns.issubset(solution.columns) or not required_columns.issubset(submission.columns):
        raise ParticipantVisibleError("Solution and submission must contain 'id' and 'segmentation' columns")
    
    # Ensure the IDs match between solution and submission
    if not solution[row_id_column_name].equals(submission[row_id_column_name]):
        raise ParticipantVisibleError("Submission IDs do not match solution IDs")
    
    # Delete the row ID column as Kaggle aligns solution and submission before passing to score()
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    # Decode RLE masks and compute Dice score
    dice_scores = []
    for solution_seg, submission_seg in zip(solution["segmentation"], submission["segmentation"]):
        solution_mask = rle_decode(solution_seg,shape=(512, 512))
        submission_mask = rle_decode(submission_seg,shape=(512, 512))
        dice_scores.append(dice_coefficient(solution_mask, submission_mask))
    print(dice_scores)
    return np.mean(dice_scores)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python evaluation_function.py solution.csv submission.csv")
        sys.exit(1)

    solution_path = sys.argv[1]
    submission_path = sys.argv[2]

    solution_df = pd.read_csv(solution_path)
    submission_df = pd.read_csv(submission_path)

    final_score = score(solution_df, submission_df,"id")
    print(f"Final Score (Dice Coefficient): {final_score:.6f}")

        
