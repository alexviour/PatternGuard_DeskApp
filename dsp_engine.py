import cv2
import numpy as np
import scipy.fft as fft


# SYMMETRY & WALLPAPER GROUP
def _ncc(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-6:
        return 0.0
    return float(np.sum(a * b) / denom)


def _seed_positions(h, w, L, n_seeds=6):
    cy, cx = h // 2, w // 2
    grid_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0), (0, 0)]
    step_y = max(L, (h - L) // 3) if h > L else 0
    step_x = max(L, (w - L) // 3) if w > L else 0
    seeds = []
    for gy, gx in grid_offsets[:n_seeds]:
        y0 = int(np.clip(cy + gy * step_y - L // 2, 0, max(0, h - L)))
        x0 = int(np.clip(cx + gx * step_x - L // 2, 0, max(0, w - L)))
        seeds.append((y0, x0))
    return seeds


def _best_local_ncc(I_gray, y0, x0, L, transform_fn, search_radius, step, M_mask=None):
    h, w = I_gray.shape
    best = -1.0
    for dy in range(-search_radius, search_radius + 1, step):
        for dx in range(-search_radius, search_radius + 1, step):
            yy, xx = y0 + dy, x0 + dx
            if yy < 0 or xx < 0 or yy + L > h or xx + L > w:
                continue
            if M_mask is not None:
                region = M_mask[yy:yy + L, xx:xx + L]
                if region.size > 0 and np.mean(region > 0) > 0.15:
                    continue
            tile = I_gray[yy:yy + L, xx:xx + L]
            transformed = transform_fn(tile)
            if transformed.shape != tile.shape:
                continue
            score = _ncc(tile, transformed)
            if score > best:
                best = score
    return best


def _classify_wallpaper_group_with_evidence(
    I_gray, a1, a2, M_mask=None, ncc_threshold=0.55, vote_fraction=0.5
):
    h, w = I_gray.shape
    L = int(np.clip(round(1.5 * min(a1, a2)), 16, min(h, w) // 2))
    if L < 8:
        return {
            "group": "p1",
            "is_square": False,
            "aspect_ratio": float(a1 / a2) if a2 else 1.0,
            "tile_side": L,
            "search_radius": 0,
            "search_step": 0,
            "ncc_threshold": ncc_threshold,
            "vote_fraction_threshold": vote_fraction,
            "operations": {},
            "decision_path": "The candidate tile was too small to test symmetry; p1 was used.",
        }
    seeds = _seed_positions(h, w, L, n_seeds=6)
    search_radius = int(max(8, min(a1, a2) / 2))
    step = max(2, search_radius // 6)

    def _scores(transform_fn):
        return [
            _best_local_ncc(I_gray, y0, x0, L, transform_fn, search_radius, step, M_mask)
            for (y0, x0) in seeds
        ]

    def _vote(scores):
        valid_scores = [score for score in scores if score > -1.0]
        passes = sum(1 for score in valid_scores if score >= ncc_threshold)
        fraction = passes / len(valid_scores) if valid_scores else 0.0
        return {
            "scores": valid_scores,
            "valid_count": len(valid_scores),
            "pass_count": passes,
            "vote_fraction": fraction,
            "passed": bool(valid_scores and fraction >= vote_fraction),
        }

    operations = {
        "mirror_h": _vote(_scores(lambda tile: np.fliplr(tile))),
        "mirror_v": _vote(_scores(lambda tile: np.flipud(tile))),
        "rotation_180": _vote(_scores(lambda tile: np.rot90(tile, 2))),
    }

    def _did_pass(operation):
        return operations[operation]["passed"]

    ratio = a1 / a2 if a2 != 0 else 1.0
    is_square = abs(ratio - 1.0) < 0.20
    has_2fold = _did_pass("rotation_180")
    has_mirror_h = _did_pass("mirror_h")
    has_mirror_v = _did_pass("mirror_v")
    has_mirror = has_mirror_h or has_mirror_v

    if is_square:
        operations["rotation_90"] = _vote(_scores(lambda tile: np.rot90(tile, 1)))
        has_4fold = _did_pass("rotation_90")
        if has_4fold and has_mirror:
            group = "p4m"
            decision_path = "Near-square lattice; 90-degree rotation and at least one mirror vote passed."
        elif has_4fold:
            group = "p4"
            decision_path = "Near-square lattice; 90-degree rotation passed and mirror votes did not."
        elif has_2fold and has_mirror:
            group = "cmm"
            decision_path = "Near-square lattice; 180-degree rotation and at least one mirror vote passed."
        elif has_mirror:
            group = "pm"
            decision_path = "Near-square lattice; at least one mirror vote passed, but the higher-symmetry tests did not."
        elif has_2fold:
            group = "p2"
            decision_path = "Near-square lattice; 180-degree rotation passed, but mirror votes did not."
        else:
            group = "p1"
            decision_path = "Near-square lattice; no tested non-trivial rotation or mirror vote passed."
    elif has_2fold and has_mirror_h and has_mirror_v:
        group = "pmm"
        decision_path = "Non-square lattice; 180-degree rotation and both horizontal and vertical mirror votes passed."
    elif has_2fold and has_mirror:
        group = "cmm"
        decision_path = "Non-square lattice; 180-degree rotation and at least one mirror vote passed, but not both mirror votes."
    elif has_mirror:
        group = "pm"
        decision_path = "Non-square lattice; at least one mirror vote passed, but the 180-degree rotation vote did not."
    elif has_2fold:
        group = "p2"
        decision_path = "Non-square lattice; 180-degree rotation passed, but mirror votes did not."
    else:
        group = "p1"
        decision_path = "Non-square lattice; no tested non-trivial rotation or mirror vote passed."

    return {
        "group": group,
        "is_square": is_square,
        "aspect_ratio": ratio,
        "tile_side": L,
        "search_radius": search_radius,
        "search_step": step,
        "ncc_threshold": ncc_threshold,
        "vote_fraction_threshold": vote_fraction,
        "operations": operations,
        "decision_path": decision_path,
    }


def classify_wallpaper_group(I_gray, a1, a2, M_mask=None, ncc_threshold=0.55, vote_fraction=0.5):
    return _classify_wallpaper_group_with_evidence(
        I_gray, a1, a2, M_mask, ncc_threshold, vote_fraction
    )["group"]


"""2D Wiener-Khinchin FFT & Wallpaper Group Classification."""


def run_algorithm_i_autocorrelation(I_gray, include_evidence=False):
    F = fft.fft2(I_gray)
    S = F * np.conj(F)
    R_xx = fft.ifft2(S)
    R_xx = np.abs(fft.fftshift(R_xx))
    h, w = I_gray.shape
    cy, cx = h // 2, w // 2

    R_xx_filtered = R_xx.copy()
    cv2.circle(R_xx_filtered, (cx, cy), 20, 0, -1)
    _, _, _, max_loc = cv2.minMaxLoc(R_xx_filtered)

    a2_val = float(abs(max_loc[0] - cx))
    a1_val = float(abs(max_loc[1] - cy))

    if a1_val < 15:
        a1_val = 48.0
    if a2_val < 15:
        a2_val = 48.0

    theta_dom = float(np.degrees(np.arctan2(a1_val, a2_val)))

    try:
        symmetry_evidence = _classify_wallpaper_group_with_evidence(
            I_gray, a1_val, a2_val
        )
        detected_group = symmetry_evidence["group"]
    except Exception as e:
        print(f"[Wallpaper Classification Warning] {e}")
        detected_group = "p1"
        symmetry_evidence = {
            "group": detected_group,
            "is_square": False,
            "aspect_ratio": a1_val / a2_val if a2_val else 1.0,
            "tile_side": 0,
            "search_radius": 0,
            "search_step": 0,
            "ncc_threshold": 0.55,
            "vote_fraction_threshold": 0.5,
            "operations": {},
            "decision_path": "Symmetry scoring failed; the classifier fell back to p1.",
        }

    result = (R_xx, a1_val, a2_val, theta_dom, detected_group)
    return result + (symmetry_evidence,) if include_evidence else result


"""Conservative defect detection."""


def run_algorithm_ii_defect_isolation(I_bgr, a1, a2):
    I_gray = cv2.cvtColor(I_bgr, cv2.COLOR_BGR2GRAY)

    # Target dark marks/holes
    _, M_dark = cv2.threshold(I_gray, 35, 255, cv2.THRESH_BINARY_INV)

    # Detect scratches
    k_size = max(15, int(min(a1, a2) // 2))
    if k_size % 2 == 0:
        k_size += 1

    local_mean = cv2.boxFilter(I_gray, -1, (k_size, k_size))
    diff_map = cv2.absdiff(I_gray, local_mean)

    _, M_structural = cv2.threshold(diff_map, 110, 255, cv2.THRESH_BINARY)
    kernel_slant1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    kernel_slant2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    M_lines = cv2.morphologyEx(M_structural, cv2.MORPH_OPEN, kernel_slant1)
    M_lines = cv2.bitwise_or(M_lines, cv2.morphologyEx(M_structural, cv2.MORPH_OPEN, kernel_slant2))

    M_mask = cv2.bitwise_or(M_dark, M_structural)
    M_mask = cv2.bitwise_or(M_mask, M_lines)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    M_mask = cv2.morphologyEx(M_mask, cv2.MORPH_OPEN, kernel_clean)
    M_mask = cv2.dilate(M_mask, kernel_clean, iterations=1)

    return M_mask, diff_map.astype(np.float32)


def _find_exact_lattice_patch(I_color, M_mask, y1, y2, x1, x2, step_y, step_x, local_comp_mask):
    """Locates optimal non-defective lattice patch via border-ring NCC match."""
    h, w, _ = I_color.shape
    patch_h, patch_w = y2 - y1, x2 - x1
    if patch_h <= 0 or patch_w <= 0:
        return None

    I_gray = cv2.cvtColor(I_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_patch = I_gray[y1:y2, x1:x2]
    ring_mask = cv2.dilate(
        local_comp_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ) > 0
    ring_mask &= local_comp_mask == 0
    if np.count_nonzero(ring_mask) <= 10:
        ring_mask = np.ones((patch_h, patch_w), dtype=bool)

    template = np.zeros((patch_h, patch_w), dtype=np.float32)
    ring_values = ref_patch[ring_mask]
    template[ring_mask] = ring_values - ring_values.mean()
    template_norm = float(np.linalg.norm(template))
    if template_norm < 1e-6:
        return None

    search_y = max(1, int(step_y))
    search_x = max(1, int(step_x))
    roi_y1 = max(0, y1 - 3 * search_y)
    roi_x1 = max(0, x1 - 3 * search_x)
    roi_y2 = min(h, y2 + 3 * search_y)
    roi_x2 = min(w, x2 + 3 * search_x)
    source = I_gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if source.shape[0] < patch_h or source.shape[1] < patch_w:
        return None

    mask_u8 = ring_mask.astype(np.uint8)
    ring_count = float(np.count_nonzero(ring_mask))
    cross_correlation = cv2.matchTemplate(
        source, template, cv2.TM_CCORR, mask=mask_u8
    )
    local_sum = cv2.matchTemplate(
        source, mask_u8.astype(np.float32), cv2.TM_CCORR
    )
    local_square_sum = cv2.matchTemplate(
        source * source, mask_u8.astype(np.float32), cv2.TM_CCORR
    )
    local_variance = np.maximum(
        local_square_sum - (local_sum * local_sum / ring_count), 0.0
    )
    scores = cross_correlation / np.maximum(
        template_norm * np.sqrt(local_variance), 1e-6
    )

    donor_validation_mask = cv2.dilate(
        local_comp_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(np.float32) / 255.0
    source_damage = (M_mask[roi_y1:roi_y2, roi_x1:roi_x2] > 0).astype(
        np.float32
    )
    contaminated_candidates = cv2.matchTemplate(
        source_damage, donor_validation_mask, cv2.TM_CCORR
    )
    score_y1 = max(0, y1 - 3 * search_y - roi_y1)
    score_x1 = max(0, x1 - 3 * search_x - roi_x1)
    score_y2 = min(scores.shape[0], y1 + 3 * search_y - roi_y1 + 1)
    score_x2 = min(scores.shape[1], x1 + 3 * search_x - roi_x1 + 1)
    search_scores = scores[score_y1:score_y2, score_x1:score_x2]
    if search_scores.size == 0:
        return None

    candidate_overlap = contaminated_candidates[
        score_y1:score_y2, score_x1:score_x2
    ]
    valid_candidates = candidate_overlap < 0.5
    target_y = y1 - roi_y1 - score_y1
    target_x = x1 - roi_x1 - score_x1
    if (
        0 <= target_y < valid_candidates.shape[0]
        and 0 <= target_x < valid_candidates.shape[1]
    ):
        valid_candidates[target_y, target_x] = False
    if not np.any(valid_candidates):
        return None

    best_index = np.argmax(
        np.where(valid_candidates, search_scores, -np.inf)
    )
    local_y, local_x = np.unravel_index(best_index, search_scores.shape)
    cy1 = roi_y1 + score_y1 + int(local_y)
    cx1 = roi_x1 + score_x1 + int(local_x)
    return I_color[cy1:cy1 + patch_h, cx1:cx1 + patch_w]


""" Lattice-Guided Fabric Reconstruction Engine."""


def run_algorithm_iii_template_alignment(I_color, M_mask, a1, a2, patch_size=24, stride=1, symmetry_group="p1"):
    h, w, _ = I_color.shape
    I_out = I_color.copy()

    if M_mask.ndim == 3:
        M_mask = M_mask[:, :, 0]

    if np.count_nonzero(M_mask) == 0:
        return I_color

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(M_mask)

    step_y = max(16, int(a1))
    step_x = max(16, int(a2))

    for label in range(1, num_labels):
        comp_mask = (labels == label).astype(np.uint8) * 255
        area = stats[label, cv2.CC_STAT_AREA]

        if area < 4:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        bw = stats[label, cv2.CC_STAT_WIDTH]
        bh = stats[label, cv2.CC_STAT_HEIGHT]

        pad = max(12, int(max(bw, bh) * 0.35))
        y1, y2 = max(0, y - pad), min(h, y + bh + pad)
        x1, x2 = max(0, x - pad), min(w, x + bw + pad)

        local_comp_mask = comp_mask[y1:y2, x1:x2]

        donor_patch = _find_exact_lattice_patch(I_color, M_mask, y1, y2, x1, x2, step_y, step_x, local_comp_mask)

        if donor_patch is not None and donor_patch.shape[:2] == (y2 - y1, x2 - x1):
            local_mask = comp_mask[y1:y2, x1:x2]
            repair_mask = cv2.dilate(
                local_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ) > 0
            target_patch = I_color[y1:y2, x1:x2].astype(np.float32)
            calibrated_donor = donor_patch.astype(np.float32)
            ring_mask = cv2.dilate(
                local_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            ) > 0
            ring_mask &= local_mask == 0
            if np.count_nonzero(ring_mask) > 10:
                for channel in range(I_color.shape[2]):
                    source_values = calibrated_donor[:, :, channel][ring_mask]
                    target_values = target_patch[:, :, channel][ring_mask]
                    source_std = float(np.std(source_values))
                    target_std = float(np.std(target_values))
                    gain = (
                        np.clip(target_std / source_std, 0.85, 1.15)
                        if source_std > 1.0
                        else 1.0
                    )
                    offset = np.clip(
                        float(np.mean(target_values))
                        - gain * float(np.mean(source_values)),
                        -24.0,
                        24.0,
                    )
                    calibrated_donor[:, :, channel] = (
                            calibrated_donor[:, :, channel] * gain + offset
                    )

            repaired_patch = np.clip(calibrated_donor, 0, 255).astype(np.uint8)
            I_out[y1:y2, x1:x2][repair_mask] = repaired_patch[repair_mask]

        else:
            # Boundary fallback
            dilated_mask = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            I_out = cv2.inpaint(I_out, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return I_out
