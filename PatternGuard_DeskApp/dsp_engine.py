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

def classify_wallpaper_group(I_gray, a1, a2, M_mask=None, ncc_threshold=0.55, vote_fraction=0.5):
   h, w = I_gray.shape
   L = int(np.clip(round(1.5 * min(a1, a2)), 16, min(h, w) // 2))
   if L < 8:
       return "p1"
   seeds = _seed_positions(h, w, L, n_seeds=6)
   search_radius = int(max(8, min(a1, a2) / 2))
   step = max(2, search_radius // 6)

   def _vote(transform_fn):
       scores = [
           _best_local_ncc(I_gray, y0, x0, L, transform_fn, search_radius, step, M_mask)
           for (y0, x0) in seeds
       ]
       scores = [s for s in scores if s > -1.0]
       if not scores:
           return False
       passes = sum(1 for s in scores if s >= ncc_threshold)
       return (passes / len(scores)) >= vote_fraction

   ratio = a1 / a2 if a2 != 0 else 1.0
   is_square = abs(ratio - 1.0) < 0.20
   has_2fold = _vote(lambda t: np.rot90(t, 2))
   has_mirror_h = _vote(lambda t: np.fliplr(t))
   has_mirror_v = _vote(lambda t: np.flipud(t))
   has_mirror = has_mirror_h or has_mirror_v

   if is_square:
       has_4fold = _vote(lambda t: np.rot90(t, 1))
       if has_4fold and has_mirror:
           return "p4m"
       if has_4fold:
           return "p4"
       if has_2fold and has_mirror:
           return "cmm"
       if has_mirror:
           return "pm"
       if has_2fold:
           return "p2"
       return "p1"

   if has_2fold and (has_mirror_h and has_mirror_v):
       return "pmm"
   if has_2fold and has_mirror:
       return "cmm"
   if has_mirror:
       return "pm"
   if has_2fold:
       return "p2"
   return "p1"

"""2D Wiener-Khinchin FFT & Wallpaper Group Classification."""
def run_algorithm_i_autocorrelation(I_gray):
   
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
       detected_group = classify_wallpaper_group(I_gray, a1_val, a2_val)
   except Exception as e:
       print(f"[Wallpaper Classification Warning] {e}")
       detected_group = "p1"

   return R_xx, a1_val, a2_val, theta_dom, detected_group

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
   patch_h = y2 - y1
   patch_w = x2 - x1

   I_gray = cv2.cvtColor(I_color, cv2.COLOR_BGR2GRAY)
   ref_patch = I_gray[y1:y2, x1:x2]

   # Border ring mask
   ring_mask = cv2.dilate(local_comp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) - local_comp_mask
   valid_ring_pts = ring_mask > 0

   best_donor = None
   best_ncc = -1.0

   for mult_y in [-3, -2, -1, 0, 1, 2, 3]:
       for mult_x in [-3, -2, -1, 0, 1, 2, 3]:
           if mult_y == 0 and mult_x == 0:
               continue

           base_dy = int(mult_y * step_y)
           base_dx = int(mult_x * step_x)

           for fine_dy in range(base_dy - 6, base_dy + 7, 2):
               for fine_dx in range(base_dx - 6, base_dx + 7, 2):
                   cy1, cy2 = y1 + fine_dy, y2 + fine_dy
                   cx1, cx2 = x1 + fine_dx, x2 + fine_dx

                   if cy1 < 0 or cx1 < 0 or cy2 > h or cx2 > w:
                       continue

                   # Ensure donor region is clean
                   if np.count_nonzero(M_mask[cy1:cy2, cx1:cx2]) > 0:
                       continue

                   candidate_patch = I_color[cy1:cy2, cx1:cx2]

                   # Avoid flat regions/ w/o patterns
                   if np.std(candidate_patch) < 6:
                       continue

                   cand_gray = I_gray[cy1:cy2, cx1:cx2]

        
                   if np.count_nonzero(valid_ring_pts) > 10:
                       score = _ncc(ref_patch[valid_ring_pts], cand_gray[valid_ring_pts])
                   else:
                       score = _ncc(ref_patch, cand_gray)

                   if score > best_ncc:
                       best_ncc = score
                       best_donor = candidate_patch

   return best_donor

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

           kernel_feather = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
           dilated_mask = cv2.dilate(local_mask, kernel_feather, iterations=1)

           alpha = cv2.GaussianBlur(dilated_mask.astype(np.float32), (3, 3), 0) / 255.0
           alpha = np.clip(alpha, 0.0, 1.0)[:, :, np.newaxis]

           bg = I_out[y1:y2, x1:x2].astype(np.float32)
           fg = donor_patch.astype(np.float32)
           blended = fg * alpha + bg * (1.0 - alpha)

           I_out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

       else:
           # Boundary fallback
           dilated_mask = cv2.dilate(comp_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
           I_out = cv2.inpaint(I_out, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

   return I_out
