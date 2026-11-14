# evaluation_metrics.py
from typing import Optional, Tuple
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def _normalize_image(arr, name):
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2D grayscale image, got shape {np.asarray(arr).shape}")
    return a.astype(np.float32)

def _normalize_mask(arr, name, threshold):
    a = np.asarray(arr)
    if a.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be a 2D mask or 3D color mask, got shape {a.shape}"
        )
    mask_scale = 255.0 if a.size and float(np.max(a)) > 1.0 else 1.0
    threshold_value = threshold * mask_scale
    if a.ndim == 3:
        a = np.any(a > threshold_value, axis=2)
    else:
        a = a > threshold_value
    return a.astype(np.uint8)


def calculate_reference_free_metrics(
    I_input: np.ndarray,
    I_reconstructed: np.ndarray,
    M_repair: np.ndarray,
    ssim_score: float,
) -> dict[str, Optional[float]]:
    """Estimate restoration plausibility without treating the input as clean ground truth.

    The returned scores are heuristics, not substitutes for labeled evaluation.
    """
    input_image = np.asarray(I_input, dtype=np.float32)
    reconstructed = np.asarray(I_reconstructed, dtype=np.float32)
    if input_image.ndim not in (2, 3) or reconstructed.ndim not in (2, 3):
        raise ValueError("Images must be grayscale or color arrays")
    if input_image.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: Input {input_image.shape} vs "
            f"Reconstructed {reconstructed.shape}"
        )

    repair = _normalize_mask(M_repair, "M_repair", 0.5).astype(bool)
    if repair.shape != input_image.shape[:2]:
        raise ValueError(
            f"Shape mismatch: M_repair {repair.shape} vs image "
            f"{input_image.shape[:2]}"
        )

    components: dict[str, Optional[float]] = {
        "outside_preservation": None,
        "boundary_continuity": None,
        "ssim": float(np.clip(ssim_score, 0.0, 100.0)),
    }
    if not np.any(repair):
        return {
            **components,
            "outside_mask_mae": None,
            "boundary_intensity_jump": None,
            "estimated_sfs": None,
        }

    if input_image.ndim == 3:
        reconstructed_gray = np.mean(reconstructed, axis=2)
        pixel_error = np.mean(
            np.abs(input_image - reconstructed), axis=2
        )
    else:
        reconstructed_gray = reconstructed
        pixel_error = np.abs(input_image - reconstructed)

    outside = ~repair
    outside_mask_mae = (
        float(np.mean(pixel_error[outside])) if np.any(outside) else None
    )
    if outside_mask_mae is not None:
        components["outside_preservation"] = float(
            100.0 * np.exp(-outside_mask_mae / 12.0)
        )

    mask_u8 = repair.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    inner_ring = repair & (cv2.erode(mask_u8, kernel) == 0)
    outer_ring = (cv2.dilate(mask_u8, kernel) > 0) & ~repair
    boundary_intensity_jump = None
    if np.any(inner_ring) and np.any(outer_ring):
        boundary_intensity_jump = float(
            abs(
                float(np.mean(reconstructed_gray[inner_ring]))
                - float(np.mean(reconstructed_gray[outer_ring]))
            )
        )
        components["boundary_continuity"] = float(
            100.0 * np.exp(-boundary_intensity_jump / 40.0)
        )

    estimated_sfs = (
        float(np.mean(list(components.values())))
        if all(value is not None for value in components.values())
        else None
    )
    return {
        **components,
        "outside_mask_mae": outside_mask_mae,
        "boundary_intensity_jump": boundary_intensity_jump,
        "estimated_sfs": estimated_sfs,
    }


def calculate_restoration_metrics(
        I_original_gray: np.ndarray,
        I_reconstructed_gray: np.ndarray,
        M_mask: np.ndarray,
        M_predicted: Optional[np.ndarray] = None,
        change_threshold: float = 12.0,
        mask_threshold: float = 0.5
) -> Tuple[
    float,
    float,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    """
    Computes SSIM, clean-reference restoration accuracy, and, when a separately
    predicted mask is supplied, defect-mask metrics:
    (ssim_value, accuracy_value, recall_value, fpr_value, iou_value, sfs_score).

    Recall, FPR, IoU, and SFS are None when ``M_predicted`` is omitted. A mask
    inferred from restoration changes is not a defect-detection prediction and
    must not be used as one.

    ``M_mask`` is the ground-truth damage mask and ``M_predicted`` is the
    predicted damage mask.

    The SFS is the weighted mean of restoration accuracy (35%), SSIM (25%), Recall
    (15%), specificity (15%), and IoU (10%). Clean-region integrity is already
    represented by restoration accuracy, SSIM, and specificity, so no additional
    clean-region penalty is applied.

    Restoration accuracy combines defect-pixel repair accuracy (60%) and
    clean-pixel integrity (40%), each counting pixels within the inclusive
    ``change_threshold`` of the clean reference. SSIM uses an 8-bit data range
    and reports the clipped 0-1 SSIM value as a percentage over clean pixels.
    The caller must pass the clean reference as ``I_original_gray`` for
    reconstruction measures to represent fidelity.

    Mask values are binarized when they exceed ``mask_threshold``. The
    threshold is relative to a 0-1 mask, or scaled to 0-255 for byte masks.
    """
    if not np.isfinite(mask_threshold) or not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be a finite value between 0 and 1")

    try:
        if I_original_gray is None or I_reconstructed_gray is None or M_mask is None:
            raise ValueError("One or more input arrays are None")

        I_orig = _normalize_image(I_original_gray, "I_original_gray")
        I_recon = _normalize_image(I_reconstructed_gray, "I_reconstructed_gray")
        if I_orig.shape != I_recon.shape:
            raise ValueError(f"Shape mismatch: Original {I_orig.shape} vs Reconstructed {I_recon.shape}")

        M_defect = _normalize_mask(M_mask, "M_mask", mask_threshold)
        if M_defect.shape != I_orig.shape:
            raise ValueError(f"Shape mismatch: M_mask {M_defect.shape} vs image {I_orig.shape}")

        if M_predicted is not None:
            M_pred = _normalize_mask(M_predicted, "M_predicted", mask_threshold)
            if M_pred.shape != I_orig.shape:
                raise ValueError(
                    f"Shape mismatch: M_predicted {M_pred.shape} vs image {I_orig.shape}"
                )
        else:
            M_pred = None

        # 1. SSIM Evaluation on clean regions
        mask_clean = (M_defect == 0)
        clean_pixels = int(np.sum(mask_clean))
        if clean_pixels > 0:
            _, ssim_map = ssim(
                I_orig,
                I_recon,
                data_range=255.0,
                full=True,
            )
            raw_ssim = float(np.mean(ssim_map[mask_clean]))
            ssim_value = float(np.clip(raw_ssim, 0.0, 1.0) * 100.0)
        else:
            ssim_value = 100.0

        # 2. Restoration Accuracy
        defect_mask_bool = (M_defect == 1)
        defect_pixels = int(np.sum(defect_mask_bool))
        if defect_pixels > 0:
            defect_diff = np.abs(I_recon[defect_mask_bool] - I_orig[defect_mask_bool])
            reconstructed_count = int(np.sum(defect_diff <= change_threshold))
            repair_accuracy = (reconstructed_count / defect_pixels) * 100.0
        else:
            repair_accuracy = 100.0

        if clean_pixels > 0:
            clean_diff = np.abs(I_recon[mask_clean] - I_orig[mask_clean])
            untouched_count = int(np.sum(clean_diff <= change_threshold))
            clean_integrity = (untouched_count / clean_pixels) * 100.0
        else:
            clean_integrity = 100.0

        accuracy_value = float(0.60 * repair_accuracy + 0.40 * clean_integrity)

        if M_pred is None:
            recall_value = fpr_value = iou_value = sfs_score = None
        else:
            # 3. Detection metrics require an independently produced mask.
            tp = int(np.sum((M_defect == 1) & (M_pred == 1)))
            fp = int(np.sum((M_defect == 0) & (M_pred == 1)))
            fn = int(np.sum((M_defect == 1) & (M_pred == 0)))

            recall_value = (
                float((tp / (tp + fn)) * 100.0)
                if (tp + fn) > 0
                else 100.0
            )
            fpr_value = (
                float((fp / clean_pixels) * 100.0)
                if clean_pixels > 0
                else 0.0
            )
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
            sfs_score = float(np.clip(sfs_score, 0.0, 100.0))

        return (
            float(np.clip(ssim_value, 0.0, 100.0)),
            float(np.clip(accuracy_value, 0.0, 100.0)),
            (
                float(np.clip(recall_value, 0.0, 100.0))
                if recall_value is not None else None
            ),
            (
                float(np.clip(fpr_value, 0.0, 100.0))
                if fpr_value is not None else None
            ),
            (
                float(np.clip(iou_value, 0.0, 100.0))
                if iou_value is not None else None
            ),
            sfs_score,
        )
    except ValueError:
        raise