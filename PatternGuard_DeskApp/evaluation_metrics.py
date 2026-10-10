# evaluation_metrics.py
from typing import Optional, Tuple
import numpy as np
from skimage.metrics import structural_similarity as ssim

def _normalize_image(arr, name):
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale image, got shape {np.asarray(arr).shape}")
    return a.astype(np.float32)

def _normalize_mask(arr, name):
    a = np.asarray(arr)
    if a.ndim == 3:
        a = np.any(a > 0, axis=2)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D mask, got shape {np.asarray(arr).shape}")
    return (a > 0).astype(np.uint8)

def calculate_restoration_metrics(
        I_original_gray: np.ndarray,
        I_reconstructed_gray: np.ndarray,
        M_mask: np.ndarray,
        M_predicted: Optional[np.ndarray] = None,
        change_threshold: float = 12.0) -> Tuple[float, float, float, float, float, float]:
    """
    Computes SSIM, restoration accuracy, and defect metrics:
    (ssim_value, accuracy_value, recall_value, fpr_value, iou_value, sfs_score)
    """
    try:
        if I_original_gray is None or I_reconstructed_gray is None or M_mask is None:
            raise ValueError("One or more input arrays are None")

        I_orig = _normalize_image(I_original_gray, "I_original_gray")
        I_recon = _normalize_image(I_reconstructed_gray, "I_reconstructed_gray")
        if I_orig.shape != I_recon.shape:
            raise ValueError(f"Shape mismatch: Original {I_orig.shape} vs Reconstructed {I_recon.shape}")

        M_defect = _normalize_mask(M_mask, "M_mask")
        if M_defect.shape != I_orig.shape:
            raise ValueError(f"Shape mismatch: M_mask {M_defect.shape} vs image {I_orig.shape}")

        if M_predicted is None:
            pixel_diff_full = np.abs(I_orig - I_recon)
            M_pred = (pixel_diff_full > change_threshold).astype(np.uint8)
        else:
            M_pred = _normalize_mask(M_predicted, "M_predicted")

        # 1. SSIM Evaluation on clean regions
        mask_clean = (M_defect == 0)
        clean_pixels = int(np.sum(mask_clean))
        if clean_pixels > 0:
            data_range = float(np.ptp(I_orig))
            if data_range <= 0.0:
                data_range = 255.0
            try:
                _, ssim_map = ssim(I_orig, I_recon, data_range=data_range, full=True)
                raw_ssim = float(np.mean(ssim_map[mask_clean]))
            except Exception:
                raw_ssim = float(ssim(I_orig, I_recon, data_range=data_range))
            ssim_value = float(np.clip((raw_ssim + 1.0) / 2.0 * 100.0, 0.0, 100.0))
        else:
            ssim_value = 100.0

        # 2. Restoration Accuracy
        defect_mask_bool = (M_defect == 1)
        defect_pixels = int(np.sum(defect_mask_bool))
        if defect_pixels > 0:
            defect_diff = np.abs(I_recon[defect_mask_bool] - I_orig[defect_mask_bool])
            reconstructed_count = int(np.sum(defect_diff >= change_threshold))
            repair_accuracy = (reconstructed_count / defect_pixels) * 100.0
        else:
            repair_accuracy = 100.0

        if clean_pixels > 0:
            clean_diff = np.abs(I_recon[mask_clean] - I_orig[mask_clean])
            untouched_count = int(np.sum(clean_diff < change_threshold))
            clean_integrity = (untouched_count / clean_pixels) * 100.0
        else:
            clean_integrity = 100.0

        accuracy_value = float(0.60 * repair_accuracy + 0.40 * clean_integrity)

        # 3. Recall / FPR / IoU
        tp = int(np.sum((M_defect == 1) & (M_pred == 1)))
        fp = int(np.sum((M_defect == 0) & (M_pred == 1)))
        fn = int(np.sum((M_defect == 1) & (M_pred == 0)))

        recall_value = float((tp / (tp + fn)) * 100.0) if (tp + fn) > 0 else 100.0
        fpr_value = float((fp / clean_pixels) * 100.0) if clean_pixels > 0 else 0.0
        union = int(np.sum((M_defect == 1) | (M_pred == 1)))
        iou_value = float((tp / union) * 100.0) if union > 0 else 100.0

        # 4. Symmetry Fidelity Score (Composite)
        sfs_score = (
            0.35 * accuracy_value
            + 0.25 * ssim_value
            + 0.15 * recall_value
            + 0.15 * (100.0 - fpr_value)
            + 0.10 * iou_value
        )

        if clean_pixels > 0:
            background_damage = float(np.mean(np.abs(I_orig[mask_clean] - I_recon[mask_clean])))
            if background_damage > 12.0:
                damage_penalty = max(0.05, 1.0 - (background_damage / 80.0))
                sfs_score *= damage_penalty

        sfs_score = float(np.clip(sfs_score, 0.0, 100.0))

        return (
            float(np.clip(ssim_value, 0.0, 100.0)),
            float(np.clip(accuracy_value, 0.0, 100.0)),
            float(np.clip(recall_value, 0.0, 100.0)),
            float(np.clip(fpr_value, 0.0, 100.0)),
            float(np.clip(iou_value, 0.0, 100.0)),
            sfs_score,
        )
    except Exception as exc:
        print(f"[calculate_restoration_metrics] error: {exc}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0