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
        return -1.0
    return float(np.sum(a * b) / denom)


def _estimate_lattice_vectors(autocorrelation):
    h, w = autocorrelation.shape
    cy, cx = h // 2, w // 2
    filtered = autocorrelation.astype(np.float32).copy()
    exclusion_radius = max(20, int(min(h, w) * 0.12))
    cv2.circle(filtered, (cx, cy), exclusion_radius, 0, -1)
    peak_value = float(np.max(filtered))
    center_value = float(autocorrelation[cy, cx])
    if peak_value <= 0 or center_value <= 0:
        return {
            "vectors": [],
            "peak_count": 0,
            "peak_strength_ratio": 0.0,
            "confidence": 0.0,
            "reliable": False,
            "decision_path": "No significant off-center autocorrelation peaks were found.",
        }

    local_maxima = filtered == cv2.dilate(
        filtered, np.ones((5, 5), dtype=np.uint8)
    )
    peak_coordinates = np.argwhere(
        local_maxima & (filtered >= peak_value * 0.20) & (filtered > 0)
    )
    candidates = []
    for y, x in peak_coordinates:
        dx, dy = int(x - cx), int(y - cy)
        if dy < 0 or (dy == 0 and dx < 0):
            dx, dy = -dx, -dy
        length = float(np.hypot(dx, dy))
        if length <= exclusion_radius:
            continue
        if any(
            abs(candidate["dx"] - dx) <= 2
            and abs(candidate["dy"] - dy) <= 2
            for candidate in candidates
        ):
            continue
        candidates.append(
            {
                "dx": dx,
                "dy": dy,
                "length": length,
                "strength": float(filtered[y, x]),
            }
        )

    candidates.sort(key=lambda item: (item["length"], -item["strength"]))
    best_pair = None
    best_pair_key = None
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1:]:
            cross = abs(first["dx"] * second["dy"] - first["dy"] * second["dx"])
            sine_angle = cross / (first["length"] * second["length"])
            if sine_angle < 0.5:
                continue
            strength_ratio = min(first["strength"], second["strength"]) / peak_value
            if strength_ratio < 0.20:
                continue
            pair_quality = min(first["strength"], second["strength"]) / np.sqrt(
                first["length"] * second["length"]
            )
            pair_key = (-pair_quality, first["length"] + second["length"])
            if best_pair_key is None or pair_key < best_pair_key:
                best_pair = (first, second)
                best_pair_key = pair_key

    if best_pair is None:
        return {
            "vectors": [],
            "peak_count": len(candidates),
            "peak_strength_ratio": peak_value / center_value,
            "confidence": 0.0,
            "reliable": False,
            "decision_path": (
                "Off-center peaks did not contain two sufficiently strong, "
                "independent lattice directions."
            ),
        }

    first, second = sorted(best_pair, key=lambda item: item["length"])
    strength_ratio = min(first["strength"], second["strength"]) / center_value
    angle = float(
        np.degrees(
            np.arccos(
                np.clip(
                    (first["dx"] * second["dx"] + first["dy"] * second["dy"])
                    / (first["length"] * second["length"]),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    confidence = float(np.clip(strength_ratio / 0.20, 0.0, 1.0))
    reliable = strength_ratio >= 0.08
    vectors = [
        {
            "dx": vector["dx"],
            "dy": vector["dy"],
            "length": vector["length"],
        }
        for vector in (first, second)
    ]
    return {
        "vectors": vectors,
        "peak_count": len(candidates),
        "peak_strength_ratio": float(strength_ratio),
        "confidence": confidence,
        "reliable": reliable,
        "basis_angle_degrees": angle,
        "decision_path": (
            "Two independent off-center autocorrelation peaks support a "
            "candidate lattice."
            if reliable
            else "The independent peak pair is too weak to support a reliable lattice."
        ),
    }


def _rectify_lattice_image(I_gray, vectors):
    basis = np.array(
        [
            [vectors[0]["dx"], vectors[1]["dx"]],
            [vectors[0]["dy"], vectors[1]["dy"]],
        ],
        dtype=np.float64,
    )
    target_basis = np.diag([vectors[0]["length"], vectors[1]["length"]])
    transform = target_basis @ np.linalg.inv(basis)
    h, w = I_gray.shape
    corners = np.array(
        [[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64
    )
    transformed_corners = corners @ transform.T
    minimum = transformed_corners.min(axis=0)
    maximum = transformed_corners.max(axis=0)
    output_size = tuple(np.maximum(np.ceil(maximum - minimum), 1).astype(int))
    affine = np.column_stack((transform, -minimum)).astype(np.float32)
    return cv2.warpAffine(
        I_gray,
        affine,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _seed_positions(h, w, L, n_seeds=6):
    cy, cx = h // 2, w // 2
    grid_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0), (0, 0)]
    step_y = max(L, (h - L) // 3) if h > L else 0
    step_x = max(L, (w - L) // 3) if w > L else 0
    seeds = []
    for gy, gx in grid_offsets[:n_seeds]:
        y0 = int(np.clip(cy + gy * step_y - L // 2, 0, max(0, h - L)))
        x0 = int(np.clip(cx + gx * step_x - L // 2, 0, max(0, w - L)))
        position = (y0, x0)
        if position not in seeds:
            seeds.append(position)
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


def _translation_symmetry_evidence(
    I_gray,
    dx,
    dy,
    tile_side,
    ncc_threshold=0.55,
    vote_fraction=0.5,
    M_mask=None,
):
    h, w = I_gray.shape
    offset_x, offset_y = int(round(dx)), int(round(dy))
    overlap_h = h - abs(offset_y)
    overlap_w = w - abs(offset_x)
    if overlap_h < tile_side or overlap_w < tile_side:
        return {
            "dx": offset_x,
            "dy": offset_y,
            "scores": [],
            "valid_count": 0,
            "pass_count": 0,
            "vote_fraction": 0.0,
            "passed": False,
            "status": "insufficient",
        }

    source_x = max(0, -offset_x)
    source_y = max(0, -offset_y)
    target_x = max(0, offset_x)
    target_y = max(0, offset_y)
    scores = []
    for y0, x0 in _seed_positions(overlap_h, overlap_w, tile_side):
        first_y, first_x = source_y + y0, source_x + x0
        second_y, second_x = target_y + y0, target_x + x0
        first = I_gray[first_y:first_y + tile_side, first_x:first_x + tile_side]
        second = I_gray[second_y:second_y + tile_side, second_x:second_x + tile_side]
        if first.shape != second.shape:
            continue
        if M_mask is not None:
            first_mask = M_mask[first_y:first_y + tile_side, first_x:first_x + tile_side]
            second_mask = M_mask[second_y:second_y + tile_side, second_x:second_x + tile_side]
            if np.mean(first_mask > 0) > 0.15 or np.mean(second_mask > 0) > 0.15:
                continue
        score = _ncc(first, second)
        if score > -1.0:
            scores.append(score)

    valid_count = len(scores)
    pass_count = sum(score >= ncc_threshold for score in scores)
    fraction = pass_count / valid_count if valid_count else 0.0
    if valid_count < 3:
        status = "insufficient"
    elif fraction >= vote_fraction:
        status = "centered"
    elif fraction <= 0.20:
        status = "primitive"
    else:
        status = "unresolved"
    return {
        "dx": offset_x,
        "dy": offset_y,
        "scores": scores,
        "valid_count": valid_count,
        "pass_count": pass_count,
        "vote_fraction": fraction,
        "passed": status == "centered",
        "status": status,
    }


def _classify_wallpaper_group_with_evidence(
    I_gray,
    a1,
    a2,
    M_mask=None,
    ncc_threshold=0.55,
    vote_fraction=0.5,
    allow_rotation_90=True,
):
    h, w = I_gray.shape
    L = int(np.clip(round(1.5 * min(a1, a2)), 16, min(h, w) // 2))
    if L < 8:
        return {
            "group": "uncertain",
            "classification_status": "uncertain",
            "is_square": False,
            "aspect_ratio": float(a1 / a2) if a2 else 1.0,
            "tile_side": L,
            "search_radius": 0,
            "search_step": 0,
            "ncc_threshold": ncc_threshold,
            "vote_fraction_threshold": vote_fraction,
            "operations": {},
            "centering_translation": {},
            "decision_path": "The candidate tile was too small to test symmetry; the result is uncertain.",
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

    minimum_valid_tiles = 3
    required_operations = list(operations)
    can_test_rotation_90 = allow_rotation_90 and abs(a1 / a2 - 1.0) < 0.20
    if can_test_rotation_90:
        operations["rotation_90"] = _vote(_scores(lambda tile: np.rot90(tile, 1)))
        required_operations.append("rotation_90")
    centering_translation = _translation_symmetry_evidence(
        I_gray,
        a1 / 2.0,
        a2 / 2.0,
        L,
        ncc_threshold,
        vote_fraction,
        M_mask,
    )
    insufficient_operations = [
        operation
        for operation in required_operations
        if operations[operation]["valid_count"] < minimum_valid_tiles
    ]
    if insufficient_operations:
        return {
            "group": "uncertain",
            "classification_status": "uncertain",
            "is_square": can_test_rotation_90,
            "aspect_ratio": a1 / a2 if a2 != 0 else 1.0,
            "tile_side": L,
            "search_radius": search_radius,
            "search_step": step,
            "ncc_threshold": ncc_threshold,
            "vote_fraction_threshold": vote_fraction,
            "minimum_valid_tiles": minimum_valid_tiles,
            "operations": operations,
            "centering_translation": centering_translation,
            "decision_path": (
                "Insufficient valid candidate tiles for "
                f"{', '.join(insufficient_operations)}; the result is uncertain."
            ),
        }

    def _did_pass(operation):
        return operations[operation]["passed"]

    ratio = a1 / a2 if a2 != 0 else 1.0
    is_square = can_test_rotation_90
    has_2fold = _did_pass("rotation_180")
    has_mirror_h = _did_pass("mirror_h")
    has_mirror_v = _did_pass("mirror_v")
    has_mirror = has_mirror_h or has_mirror_v

    def _centered_family_result():
        translation_status = centering_translation["status"]
        if translation_status == "centered":
            return (
                "cmm",
                "classified",
                "Two-fold rotation and mirror symmetry passed; the half-cell "
                "translation supports a centered lattice (cmm).",
            )
        if translation_status == "primitive":
            return (
                "pmm",
                "classified",
                "Two-fold rotation and mirror symmetry passed; the half-cell "
                "translation was not detected, supporting a primitive "
                "rectangular lattice (pmm).",
            )
        return (
            "pmm/cmm",
            "partial",
            "Two-fold rotation and mirror symmetry passed, but centering-"
            "translation evidence was insufficient to distinguish pmm from cmm.",
        )

    if is_square:
        has_4fold = _did_pass("rotation_90")
        if has_4fold and not has_2fold:
            group = "uncertain"
            classification_status = "uncertain"
            decision_path = (
                "The 90-degree rotation vote passed but the required 180-degree "
                "rotation check failed; the result is uncertain."
            )
        elif has_4fold and has_mirror:
            group = "p4m"
            classification_status = "classified"
            decision_path = "Near-square lattice; 90-degree and 180-degree rotation passed with at least one mirror vote."
        elif has_4fold:
            group = "p4"
            classification_status = "classified"
            decision_path = "Near-square lattice; 90-degree rotation passed and mirror votes did not."
        elif has_2fold and has_mirror:
            group, classification_status, decision_path = _centered_family_result()
        elif has_mirror_h and has_mirror_v:
            group = "uncertain"
            classification_status = "uncertain"
            decision_path = (
                "Both mirror votes passed without the expected 180-degree "
                "rotation; the symmetry evidence is inconsistent."
            )
        elif has_mirror:
            group = "pm"
            classification_status = "classified"
            decision_path = "Near-square lattice; at least one mirror vote passed, but the higher-symmetry tests did not."
        elif has_2fold:
            group = "p2"
            classification_status = "classified"
            decision_path = "Near-square lattice; 180-degree rotation passed, but mirror votes did not."
        else:
            group = "p1"
            classification_status = "classified"
            decision_path = "Near-square lattice; no tested non-trivial rotation or mirror vote passed."
    elif has_2fold and has_mirror:
        group, classification_status, decision_path = _centered_family_result()
    elif has_mirror_h and has_mirror_v:
        group = "uncertain"
        classification_status = "uncertain"
        decision_path = (
            "Both mirror votes passed without the expected 180-degree "
            "rotation; the symmetry evidence is inconsistent."
        )
    elif has_mirror:
        group = "pm"
        classification_status = "classified"
        decision_path = "Non-square lattice; at least one mirror vote passed, but the 180-degree rotation vote did not."
    elif has_2fold:
        group = "p2"
        classification_status = "classified"
        decision_path = "Non-square lattice; 180-degree rotation passed, but mirror votes did not."
    else:
        group = "p1"
        classification_status = "classified"
        decision_path = "Non-square lattice; no tested non-trivial rotation or mirror vote passed."

    return {
        "group": group,
        "classification_status": classification_status,
        "is_square": is_square,
        "aspect_ratio": ratio,
        "tile_side": L,
        "search_radius": search_radius,
        "search_step": step,
        "ncc_threshold": ncc_threshold,
        "vote_fraction_threshold": vote_fraction,
        "minimum_valid_tiles": minimum_valid_tiles,
        "operations": operations,
        "centering_translation": centering_translation,
        "decision_path": decision_path,
    }


def classify_wallpaper_group(I_gray, a1, a2, M_mask=None, ncc_threshold=0.55, vote_fraction=0.5):
    return _classify_wallpaper_group_with_evidence(
        I_gray, a1, a2, M_mask, ncc_threshold, vote_fraction
    )["group"]


"""2D Wiener-Khinchin FFT & Wallpaper Group Classification."""


def estimate_repeat_spacing(I_gray, include_lattice=False):
    centered_image = I_gray.astype(np.float32) - float(np.mean(I_gray))
    F = fft.fft2(centered_image)
    S = F * np.conj(F)
    R_xx = fft.ifft2(S)
    R_xx = np.abs(fft.fftshift(R_xx))
    h, w = I_gray.shape
    cy, cx = h // 2, w // 2

    lattice_evidence = _estimate_lattice_vectors(R_xx)
    R_xx_filtered = R_xx.copy()
    cv2.circle(R_xx_filtered, (cx, cy), 20, 0, -1)
    _, _, _, max_loc = cv2.minMaxLoc(R_xx_filtered)

    a2_val = float(abs(max_loc[0] - cx))
    a1_val = float(abs(max_loc[1] - cy))

    if lattice_evidence["reliable"]:
        first, second = lattice_evidence["vectors"]
        a1_val, a2_val = first["length"], second["length"]
        theta_dom = float(np.degrees(np.arctan2(first["dy"], first["dx"])))
    else:
        if a1_val < 15:
            a1_val = 48.0
        if a2_val < 15:
            a2_val = 48.0
        theta_dom = float(np.degrees(np.arctan2(a1_val, a2_val)))

    result = (R_xx, a1_val, a2_val, theta_dom)
    return result + (lattice_evidence,) if include_lattice else result


def run_algorithm_i_autocorrelation(I_gray, include_evidence=False):
    R_xx, a1_val, a2_val, theta_dom, lattice_evidence = estimate_repeat_spacing(
        I_gray, include_lattice=True
    )

    try:
        orthogonal_lattice = abs(
            lattice_evidence.get("basis_angle_degrees", 0.0) - 90.0
        ) <= 15.0
        if lattice_evidence["reliable"]:
            lattice_image = _rectify_lattice_image(
                I_gray, lattice_evidence["vectors"]
            )
            allow_rotation_90 = (
                orthogonal_lattice
                and abs(a1_val / a2_val - 1.0) < 0.20
            )
            symmetry_evidence = _classify_wallpaper_group_with_evidence(
                lattice_image,
                a1_val,
                a2_val,
                allow_rotation_90=allow_rotation_90,
            )
            detected_group = symmetry_evidence["group"]
        else:
            detected_group = "uncertain"
            symmetry_evidence = {
                "group": detected_group,
                "classification_status": "uncertain",
                "is_square": False,
                "aspect_ratio": a1_val / a2_val if a2_val else 1.0,
                "tile_side": 0,
                "search_radius": 0,
                "search_step": 0,
                "ncc_threshold": 0.55,
                "vote_fraction_threshold": 0.5,
                "operations": {},
                "decision_path": lattice_evidence["decision_path"],
            }
        symmetry_evidence["lattice"] = lattice_evidence
    except Exception as e:
        print(f"[Wallpaper Classification Warning] {e}")
        detected_group = "uncertain"
        symmetry_evidence = {
            "group": detected_group,
            "classification_status": "uncertain",
            "is_square": False,
            "aspect_ratio": a1_val / a2_val if a2_val else 1.0,
            "tile_side": 0,
            "search_radius": 0,
            "search_step": 0,
            "ncc_threshold": 0.55,
            "vote_fraction_threshold": 0.5,
            "operations": {},
            "decision_path": "Symmetry scoring failed; the result is uncertain.",
            "lattice": lattice_evidence,
        }

    result = (R_xx, a1_val, a2_val, theta_dom, detected_group)
    return result + (symmetry_evidence,) if include_evidence else result


"""defect detection."""


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


def _find_exact_lattice_patch(
    I_color,
    M_mask,
    y1,
    y2,
    x1,
    x2,
    step_y,
    step_x,
    local_comp_mask,
    return_candidates=False,
    candidate_limit=8,
    lattice_basis=None,
    lattice_phase_tolerance=0.16,
):
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

    if return_candidates:
        flattened_scores = np.where(
            valid_candidates, search_scores, -np.inf
        ).ravel()
        ordered_indices = np.argsort(flattened_scores)[::-1]
        minimum_separation = max(4, min(patch_h, patch_w) // 4)
        candidate_positions = []
        candidates = []
        registered_candidates = []
        inverse_basis = (
            np.linalg.inv(lattice_basis)
            if lattice_basis is not None
            else None
        )
        for flat_index in ordered_indices:
            if (
                inverse_basis is None
                and len(candidates) >= candidate_limit
            ) or (
                inverse_basis is not None
                and len(registered_candidates) >= candidate_limit
            ) or not np.isfinite(
                flattened_scores[flat_index]
            ):
                break
            local_y, local_x = np.unravel_index(
                flat_index, search_scores.shape
            )
            if any(
                abs(local_y - selected_y) < minimum_separation
                and abs(local_x - selected_x) < minimum_separation
                for selected_y, selected_x in candidate_positions
            ):
                continue
            candidate_positions.append((local_y, local_x))
            cy1 = roi_y1 + score_y1 + int(local_y)
            cx1 = roi_x1 + score_x1 + int(local_x)
            candidate = {
                "patch": I_color[
                    cy1:cy1 + patch_h, cx1:cx1 + patch_w
                ],
                "origin": [int(cx1), int(cy1)],
                "lattice_phase_anchor": [int(x1), int(y1)],
                "search_ncc": float(flattened_scores[flat_index]),
                "lattice_registered": False,
                "lattice_cell_shift": None,
                "lattice_phase_error": None,
            }
            if inverse_basis is not None:
                displacement = np.array(
                    [cx1 - x1, cy1 - y1], dtype=np.float64
                )
                cell_coordinates = inverse_basis @ displacement
                cell_shift = np.rint(cell_coordinates).astype(np.int64)
                phase_residual = cell_coordinates - cell_shift
                phase_error = float(np.linalg.norm(phase_residual))
                if phase_error <= lattice_phase_tolerance:
                    candidate.update(
                        {
                            "lattice_registered": True,
                            "lattice_cell_shift": [
                                int(cell_shift[0]),
                                int(cell_shift[1]),
                            ],
                            "lattice_phase_error": phase_error,
                        }
                    )
                    registered_candidates.append(candidate)
            if len(candidates) < candidate_limit:
                candidates.append(candidate)
            if (
                len(candidates) >= candidate_limit
                and len(registered_candidates) >= candidate_limit
            ):
                break
        return registered_candidates or candidates or None

    best_index = np.argmax(
        np.where(valid_candidates, search_scores, -np.inf)
    )
    local_y, local_x = np.unravel_index(best_index, search_scores.shape)
    cy1 = roi_y1 + score_y1 + int(local_y)
    cx1 = roi_x1 + score_x1 + int(local_x)
    return I_color[cy1:cy1 + patch_h, cx1:cx1 + patch_w]


def _wallpaper_lattice_basis(symmetry_evidence):
    if not isinstance(symmetry_evidence, dict):
        return None
    lattice_evidence = symmetry_evidence.get("lattice", {})
    lattice_vectors = (
        lattice_evidence.get("vectors", [])
        if isinstance(lattice_evidence, dict)
        else []
    )
    if (
        not isinstance(lattice_evidence, dict)
        or not lattice_evidence.get("reliable", False)
        or not isinstance(lattice_vectors, (list, tuple))
        or len(lattice_vectors) != 2
    ):
        return None
    try:
        basis = np.array(
            [
                [lattice_vectors[0]["dx"], lattice_vectors[1]["dx"]],
                [lattice_vectors[0]["dy"], lattice_vectors[1]["dy"]],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(basis)) or abs(float(np.linalg.det(basis))) < 1e-6:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return basis


def _wallpaper_symmetry_operations(symmetry_group, symmetry_evidence):
    group_operations = {
        "p2": {"rotation_180"},
        "pm": {"mirror_h", "mirror_v"},
        "pmm/cmm": {"mirror_h", "mirror_v", "rotation_180"},
        "pmm": {"mirror_h", "mirror_v", "rotation_180"},
        "cmm": {"mirror_h", "mirror_v", "rotation_180"},
        "p4": {"rotation_90", "rotation_180"},
        "p4m": {"mirror_h", "mirror_v", "rotation_90", "rotation_180"},
    }
    operation_matrices = {
        "mirror_h": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "mirror_v": np.array([[1.0, 0.0], [0.0, -1.0]]),
        "rotation_90": np.array([[0.0, -1.0], [1.0, 0.0]]),
        "rotation_180": np.array([[-1.0, 0.0], [0.0, -1.0]]),
    }
    if not isinstance(symmetry_evidence, dict):
        return [], "symmetry_evidence_missing", None

    if symmetry_evidence.get("classification_status") not in {
        "classified",
        "partial",
    }:
        return [], "wallpaper_group_uncertain", None
    evidence_operations = symmetry_evidence.get("operations", {})
    basis = _wallpaper_lattice_basis(symmetry_evidence)
    if basis is None:
        lattice = symmetry_evidence.get("lattice", {})
        if not isinstance(lattice, dict) or not lattice.get("reliable", False):
            return [], "lattice_unreliable", None
        return [], "lattice_basis_invalid", None
    if not isinstance(evidence_operations, dict):
        return [], "symmetry_operation_evidence_missing", basis
    inverse_basis = np.linalg.inv(basis)

    allowed_operations = group_operations.get(symmetry_group, set())
    if not allowed_operations:
        return [], "wallpaper_group_has_no_supported_reconstruction_operations", basis

    operations = []
    for operation_name, operation in operation_matrices.items():
        operation_evidence = evidence_operations.get(operation_name, {})
        if (
            operation_name not in allowed_operations
            or not isinstance(operation_evidence, dict)
            or not operation_evidence.get("passed", False)
        ):
            continue
        image_matrix = basis @ operation @ inverse_basis
        if not np.allclose(
            image_matrix.T @ image_matrix, np.eye(2), atol=0.15
        ):
            continue
        operations.append((operation_name, image_matrix))
    if not operations:
        return [], "no_group_supported_operation_passed", basis
    return operations, None, basis


def _wallpaper_symmetry_matrices(symmetry_group, symmetry_evidence):
    operations, _, _ = _wallpaper_symmetry_operations(
        symmetry_group, symmetry_evidence
    )
    return [matrix for _, matrix in operations]


def _masked_patch_ncc(first, second, mask):
    first_values = first[mask].astype(np.float64)
    second_values = second[mask].astype(np.float64)
    if first_values.size < 10:
        return -1.0
    first_values -= first_values.mean()
    second_values -= second_values.mean()
    denominator = np.linalg.norm(first_values) * np.linalg.norm(second_values)
    if denominator < 1e-6:
        return -1.0
    return float(np.dot(first_values, second_values) / denominator)


def _select_symmetry_equivalent_patch(
    donor_patch,
    target_patch,
    ring_mask,
    symmetry_matrices,
    lattice_basis,
    return_details=False,
    minimum_improvement=0.03,
):
    details = {
        "applied": False,
        "operation": None,
        "lattice_center_offset": None,
        "baseline_ncc": None,
        "best_candidate_ncc": None,
        "selected_ncc": None,
        "improvement": None,
        "reason": "no_symmetry_candidate",
    }

    def _result(patch):
        return (patch, details) if return_details else patch

    if not symmetry_matrices or not np.any(ring_mask):
        details["reason"] = (
            "no_supported_symmetry_operation"
            if not symmetry_matrices
            else "insufficient_undamaged_boundary"
        )
        return _result(donor_patch)

    target_gray = cv2.cvtColor(target_patch, cv2.COLOR_BGR2GRAY)
    donor_gray = cv2.cvtColor(donor_patch, cv2.COLOR_BGR2GRAY)
    baseline_score = _masked_patch_ncc(target_gray, donor_gray, ring_mask)
    details["baseline_ncc"] = baseline_score
    details["best_candidate_ncc"] = baseline_score
    details["selected_ncc"] = baseline_score
    details["improvement"] = 0.0
    details["reason"] = "improvement_below_required_margin"
    best_patch = donor_patch
    best_score = baseline_score
    height, width = donor_patch.shape[:2]
    patch_center = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
    center_offsets = (
        lattice_basis
        @ np.array(
            [
                [x_fraction, y_fraction]
                for y_fraction in (-0.5, -0.25, 0.0, 0.25)
                for x_fraction in (-0.5, -0.25, 0.0, 0.25)
            ],
            dtype=np.float64,
        ).T
    ).T
    padding = max(height, width)
    padded_donor = cv2.copyMakeBorder(
        donor_patch,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_REFLECT_101,
    )
    padded_center_offset = np.array([padding, padding], dtype=np.float64)

    for center_offset in center_offsets:
        center = patch_center + center_offset
        padded_center = padded_center_offset + center
        for operation in symmetry_matrices:
            if isinstance(operation, tuple):
                operation_name, linear = operation
            else:
                operation_name, linear = "unknown", operation
            affine_offset = padded_center - linear @ padded_center
            affine = np.column_stack((linear, affine_offset)).astype(np.float32)
            transformed_padded = cv2.warpAffine(
                padded_donor,
                affine,
                (padded_donor.shape[1], padded_donor.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            transformed = transformed_padded[
                padding:padding + height, padding:padding + width
            ]
            transformed_gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
            score = _masked_patch_ncc(target_gray, transformed_gray, ring_mask)
            if score > best_score:
                best_patch = transformed
                best_score = score
                details.update(
                    {
                        "operation": operation_name,
                        "best_candidate_ncc": score,
                        "improvement": score - baseline_score,
                        "lattice_center_offset": [
                            float(center_offset[0]),
                            float(center_offset[1]),
                        ],
                    }
                )
    if best_score < baseline_score + minimum_improvement:
        return _result(donor_patch)
    details.update(
        {
            "applied": True,
            "selected_ncc": best_score,
            "improvement": best_score - baseline_score,
            "reason": None,
        }
    )
    return _result(best_patch)


def _select_symmetry_donor_candidate(
    candidates,
    target_patch,
    ring_mask,
    symmetry_operations,
    lattice_basis,
):
    normalized_candidates = [
        candidate
        if isinstance(candidate, dict)
        else {"patch": candidate, "origin": None}
        for candidate in candidates
    ]
    target_gray = cv2.cvtColor(target_patch, cv2.COLOR_BGR2GRAY)
    baseline_candidate = max(
        normalized_candidates,
        key=lambda candidate: _masked_patch_ncc(
            target_gray,
            cv2.cvtColor(candidate["patch"], cv2.COLOR_BGR2GRAY),
            ring_mask,
        ),
    )
    baseline_patch = baseline_candidate["patch"]
    baseline_ncc = _masked_patch_ncc(
        target_gray,
        cv2.cvtColor(baseline_patch, cv2.COLOR_BGR2GRAY),
        ring_mask,
    )
    details = {
        "applied": False,
        "operation": None,
        "lattice_center_offset": None,
        "donor_origin": baseline_candidate.get("origin"),
        "lattice_phase_anchor": baseline_candidate.get(
            "lattice_phase_anchor"
        ),
        "lattice_registered": baseline_candidate.get(
            "lattice_registered", False
        ),
        "lattice_cell_shift": baseline_candidate.get(
            "lattice_cell_shift"
        ),
        "lattice_phase_error": baseline_candidate.get(
            "lattice_phase_error"
        ),
        "baseline_ncc": baseline_ncc,
        "best_candidate_ncc": baseline_ncc,
        "selected_ncc": baseline_ncc,
        "improvement": 0.0,
        "reason": "improvement_below_required_margin",
    }
    registered_candidates = [
        candidate
        for candidate in normalized_candidates
        if candidate.get("lattice_registered", False)
    ]
    if not registered_candidates:
        details["reason"] = "no_lattice_registered_donor_candidate"
        return baseline_patch, details
    if not symmetry_operations or not np.any(ring_mask):
        details["reason"] = (
            "no_supported_symmetry_operation"
            if not symmetry_operations
            else "insufficient_undamaged_boundary"
        )
        return baseline_patch, details

    best_transformed_patch = None
    best_transformed_ncc = baseline_ncc
    best_transformed_details = None
    best_transformed_candidate = None
    for candidate in registered_candidates:
        transformed_patch, transformed_details = (
            _select_symmetry_equivalent_patch(
                candidate["patch"],
                target_patch,
                ring_mask,
                symmetry_operations,
                lattice_basis,
                return_details=True,
                minimum_improvement=0.0,
            )
        )
        candidate_ncc = transformed_details["best_candidate_ncc"]
        if (
            transformed_details["applied"]
            and candidate_ncc is not None
            and candidate_ncc > best_transformed_ncc
        ):
            best_transformed_patch = transformed_patch
            best_transformed_ncc = candidate_ncc
            best_transformed_details = transformed_details
            best_transformed_candidate = candidate

    if (
        best_transformed_patch is None
        or best_transformed_ncc < baseline_ncc + 0.03
    ):
        return baseline_patch, details

    details.update(
        {
            "applied": True,
            "operation": best_transformed_details["operation"],
            "lattice_center_offset": best_transformed_details[
                "lattice_center_offset"
            ],
            "donor_origin": best_transformed_candidate.get("origin"),
            "lattice_phase_anchor": best_transformed_candidate.get(
                "lattice_phase_anchor"
            ),
            "lattice_registered": True,
            "lattice_cell_shift": best_transformed_candidate.get(
                "lattice_cell_shift"
            ),
            "lattice_phase_error": best_transformed_candidate.get(
                "lattice_phase_error"
            ),
            "best_candidate_ncc": best_transformed_ncc,
            "selected_ncc": best_transformed_ncc,
            "improvement": best_transformed_ncc - baseline_ncc,
            "reason": None,
        }
    )
    return best_transformed_patch, details


""" Lattice-Guided Fabric Reconstruction Engine."""


def run_algorithm_iii_template_alignment(
    I_color,
    M_mask,
    a1,
    a2,
    patch_size=24,
    stride=1,
    symmetry_group="p1",
    symmetry_evidence=None,
    include_symmetry_evidence=False,
):
    h, w, _ = I_color.shape
    I_out = I_color.copy()

    if M_mask.ndim == 3:
        M_mask = M_mask[:, :, 0]

    if np.count_nonzero(M_mask) == 0:
        if include_symmetry_evidence:
            operations, reason, _ = _wallpaper_symmetry_operations(
                symmetry_group, symmetry_evidence
            )
            status = (
                "qualified_no_transform_used"
                if operations
                else "not_qualified"
            )
            return I_color, {
                "status": status,
                "qualified": bool(operations),
                "qualification_reason": reason,
                "reason": "no_repair_regions",
                "eligible_operations": [
                    name for name, _ in operations
                ],
                "guided_region_count": 0,
                "fallback_region_count": 0,
                "regions": [],
            }
        return I_color

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(M_mask)

    step_y = max(16, int(a1))
    step_x = max(16, int(a2))
    symmetry_operations, qualification_reason, lattice_basis = _wallpaper_symmetry_operations(
        symmetry_group, symmetry_evidence
    )
    symmetry_result = {
        "status": "not_qualified",
        "qualified": bool(symmetry_operations),
        "qualification_reason": qualification_reason,
        "reason": qualification_reason,
        "eligible_operations": [
            name for name, _ in symmetry_operations
        ],
        "guided_region_count": 0,
        "fallback_region_count": 0,
        "regions": [],
    }

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
        region_result = {
            "component": int(label),
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "repair_method": None,
            "symmetry": None,
        }

        donor_search_result = _find_exact_lattice_patch(
            I_color,
            M_mask,
            y1,
            y2,
            x1,
            x2,
            step_y,
            step_x,
            local_comp_mask,
            return_candidates=bool(symmetry_operations),
            lattice_basis=(
                lattice_basis if symmetry_operations else None
            ),
        )
        if symmetry_operations and donor_search_result is not None:
            donor_candidates = (
                donor_search_result
                if isinstance(donor_search_result, list)
                else [donor_search_result]
            )
            donor_patch = None
        else:
            donor_candidates = []
            donor_patch = donor_search_result

        has_donor = (
            bool(donor_candidates)
            if symmetry_operations
            else donor_patch is not None
            and donor_patch.shape[:2] == (y2 - y1, x2 - x1)
        )
        if has_donor:
            local_mask = comp_mask[y1:y2, x1:x2]
            repair_mask = cv2.dilate(
                local_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            ) > 0
            ring_mask = cv2.dilate(
                local_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            ) > 0
            ring_mask &= M_mask[y1:y2, x1:x2] == 0
            if symmetry_operations:
                donor_patch, symmetry_selection = (
                    _select_symmetry_donor_candidate(
                        donor_candidates,
                        I_color[y1:y2, x1:x2],
                        ring_mask,
                        symmetry_operations,
                        lattice_basis,
                    )
                )
            else:
                donor_patch, symmetry_selection = (
                    _select_symmetry_equivalent_patch(
                        donor_patch,
                        I_color[y1:y2, x1:x2],
                        ring_mask,
                        symmetry_operations,
                        lattice_basis,
                        return_details=True,
                    )
                )
            region_result["symmetry"] = symmetry_selection
            target_patch = I_color[y1:y2, x1:x2].astype(np.float32)
            calibrated_donor = donor_patch.astype(np.float32)
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
            region_result["repair_method"] = (
                "symmetry_donor"
                if symmetry_selection["applied"]
                else "donor"
            )

        else:
            # Boundary fallback
            dilated_mask = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            I_out = cv2.inpaint(I_out, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            region_result["repair_method"] = "telea"
            region_result["symmetry"] = {
                "applied": False,
                "operation": None,
                "lattice_center_offset": None,
                "baseline_ncc": None,
                "best_candidate_ncc": None,
                "selected_ncc": None,
                "improvement": None,
                "reason": "no_clean_donor_found",
            }

        symmetry_result["regions"].append(region_result)
        if region_result["repair_method"] == "symmetry_donor":
            symmetry_result["guided_region_count"] += 1
        else:
            symmetry_result["fallback_region_count"] += 1

    if not symmetry_operations:
        symmetry_result["status"] = "not_qualified"
    elif symmetry_result["guided_region_count"] == 0:
        symmetry_result["status"] = "qualified_no_transform_used"
        if symmetry_result["regions"]:
            reasons = [
                region["symmetry"]["reason"]
                for region in symmetry_result["regions"]
                if region["symmetry"] is not None
            ]
            symmetry_result["reason"] = (
                reasons[0] if reasons else "no_transform_selected"
            )
        else:
            symmetry_result["reason"] = "no_repair_regions"
    elif symmetry_result["fallback_region_count"]:
        symmetry_result["status"] = "mixed"
        symmetry_result["reason"] = "some_regions_used_fallback_reconstruction"
    else:
        symmetry_result["status"] = "symmetry_guided"
        symmetry_result["reason"] = "all_repaired_regions_used_symmetry_donors"

    return (I_out, symmetry_result) if include_symmetry_evidence else I_out
