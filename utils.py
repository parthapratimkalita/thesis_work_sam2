"""
Utility data structures and helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BBox
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class BBox:
    """Axis‑aligned bounding box (x1, y1) → (x2, y2)."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> int:
        return abs(self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    def scaled(self, factor: float,
               img_shape: Optional[Tuple[int, ...]] = None) -> "BBox":
        cx, cy = self.center
        half_w = self.width * factor / 2.0
        half_h = self.height * factor / 2.0
        x1, y1 = int(cx - half_w), int(cy - half_h)
        x2, y2 = int(cx + half_w), int(cy + half_h)
        if img_shape is not None:
            h, w = img_shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
        return BBox(x1, y1, x2, y2)

    def __repr__(self) -> str:
        return (f"BBox(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2} "
                f"| {self.width}×{self.height})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ScaleCommand
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ScaleCommand:
    """Parsed scale instruction with independent W/H factors."""
    w_factor: float     # width multiplier
    h_factor: float     # height multiplier

    def __repr__(self) -> str:
        if self.w_factor == self.h_factor:
            return f"ScaleCommand(uniform={self.w_factor:.3f}×)"
        return f"ScaleCommand(w={self.w_factor:.3f}×, h={self.h_factor:.3f}×)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scale parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_scale_input(text: str) -> Optional[ScaleCommand]:
    """
    Parse user input into width and height scale factors.

    "Nx bigger"  → factor = 1 + N    (additive: "1x bigger" = double)
    "Nx smaller" → factor = 1/(1+N)  (additive: "1x smaller" = half)
    Plain number → used directly as multiplier

    Examples:
      "1x bigger"   → factor 2.0   (double)
      "2x bigger"   → factor 3.0   (triple)
      "1x smaller"  → factor 0.5   (half)
      "2x smaller"  → factor 0.33  (third)
      "2"           → factor 2.0   (direct multiplier)
      "0.5"         → factor 0.5   (direct multiplier)
    """
    text = text.strip().lower()

    # ── independent: "w2 h3" ──
    m = re.match(r"w\s*(\d+(?:\.\d+)?)\s+h\s*(\d+(?:\.\d+)?)", text)
    if m:
        wf, hf = float(m.group(1)), float(m.group(2))
        if wf > 0 and hf > 0:
            return ScaleCommand(wf, hf)

    # ── width only ──
    m = re.match(r"(\d+(?:\.\d+)?)\s*x?\s*wider", text)
    if m:
        n = float(m.group(1))
        return ScaleCommand(1.0 + n, 1.0)

    m = re.match(r"(\d+(?:\.\d+)?)\s*x?\s*narrower", text)
    if m:
        n = float(m.group(1))
        f = 1.0 / (1.0 + n) if (1.0 + n) != 0 else 1.0
        return ScaleCommand(f, 1.0)

    # ── height only ──
    m = re.match(r"(\d+(?:\.\d+)?)\s*x?\s*taller", text)
    if m:
        n = float(m.group(1))
        return ScaleCommand(1.0, 1.0 + n)

    m = re.match(r"(\d+(?:\.\d+)?)\s*x?\s*shorter", text)
    if m:
        n = float(m.group(1))
        f = 1.0 / (1.0 + n) if (1.0 + n) != 0 else 1.0
        return ScaleCommand(1.0, f)

    # ── proportional bigger: "Nx bigger" → factor = 1 + N ──
    m = re.match(r"(\d+(?:\.\d+)?)\s*x?\s*(?:bigger|larger)", text)
    if m:
        n = float(m.group(1))
        f = 1.0 + n
        return ScaleCommand(f, f)

    # ── proportional smaller: "Nx smaller" → factor = 1/(1+N) ──
    m = re.match(r"(\d+(?:\.\d+)?)\s*x?\s*(?:smaller)", text)
    if m:
        n = float(m.group(1))
        f = 1.0 / (1.0 + n) if (1.0 + n) != 0 else None
        return ScaleCommand(f, f) if f else None

    # ── plain number: direct multiplier ──
    try:
        val = float(text.rstrip("x"))
        return ScaleCommand(val, val) if val > 0 else None
    except ValueError:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Mask → BBox
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mask_to_bbox(mask: np.ndarray) -> BBox:
    bool_mask = mask.astype(bool)
    ys, xs = np.where(bool_mask)
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def clean_mask(mask: np.ndarray,
               min_area_ratio: float = 0.005,
               morph_radius: int = 3) -> np.ndarray:
    """
    Clean a binary mask in two stages:

    1. **Morphological opening** (erode → dilate) to break thin bridges
       and trim narrow protrusions that SAM2 sometimes produces along
       the bounding-box edges.
    2. **Connected-component filtering** to remove any small blobs whose
       area is less than ``min_area_ratio`` of the largest component.

    Parameters
    ----------
    mask : (H, W) bool / uint8 array
    min_area_ratio : float
        Components smaller than  largest_area × min_area_ratio  are removed.
    morph_radius : int
        Radius of the elliptical kernel used for morphological opening.
        Larger values remove thicker artifacts but may nibble at thin
        legitimate features (leaf tips, etc.).  Default 3 is conservative.

    Returns
    -------
    Cleaned boolean mask (H, W).
    """
    uint_mask = (mask.astype(bool).astype(np.uint8)) * 255

    # Stage 1 — morphological opening: breaks thin bridges / edge wisps
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_radius * 2 + 1, morph_radius * 2 + 1)
    )
    opened = cv2.morphologyEx(uint_mask, cv2.MORPH_OPEN, kernel)

    # Stage 2 — remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )

    if num_labels <= 1:          # 0 = background only
        return mask.astype(bool)

    # stats columns: x, y, w, h, area   — label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]          # skip bg
    largest_area = areas.max()
    threshold = largest_area * min_area_ratio

    cleaned = np.zeros_like(uint_mask, dtype=bool)
    for label_id in range(1, num_labels):         # skip bg (0)
        if stats[label_id, cv2.CC_STAT_AREA] >= threshold:
            cleaned[labels == label_id] = True

    return cleaned


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Visual helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int] = (30, 220, 80),
                 alpha: float = 0.45) -> np.ndarray:
    vis = image_rgb.copy().astype(np.float32)
    bool_mask = mask.astype(bool)
    vis[bool_mask] = (vis[bool_mask] * (1 - alpha)
                      + np.array(color, dtype=np.float32) * alpha)
    return vis.astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Inpainting helpers (private)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _dilate_mask(
    mask_uint8: np.ndarray,
    radius: int,
    iterations: int,
) -> np.ndarray:
    """Expand the mask outward to cover edge residuals."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2, radius * 2)
    )
    dilated = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    return dilated


def _inpaint_basic(
    image_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Single-pass inpainting using Navier-Stokes."""
    return cv2.inpaint(image_bgr, mask_uint8, radius * 2, cv2.INPAINT_NS)


def _inpaint_multipass(
    image_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    radius: int,
    passes: int = 3,
) -> np.ndarray:
    """
    Multiple inpainting passes with shrinking radius.
    Each pass cleans up residuals from the previous one.
    """
    result = image_bgr.copy()
    current_mask = mask_uint8.copy()

    for i in range(passes):
        r = max(3, radius * 2 - i * (radius // passes))

        if i % 2 == 0:
            result = cv2.inpaint(result, current_mask, r, cv2.INPAINT_NS)
        else:
            result = cv2.inpaint(result, current_mask, r, cv2.INPAINT_TELEA)

        if i < passes - 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            current_mask = cv2.erode(current_mask, kernel, iterations=1)
            if current_mask.sum() == 0:
                break

    return result


def _feather_edges(
    original_bgr: np.ndarray,
    inpainted_bgr: np.ndarray,
    mask_uint8: np.ndarray,
    feather_px: int = 15,
) -> np.ndarray:
    """Smooth transition between inpainted area and original image."""
    blend_mask = mask_uint8.astype(np.float32) / 255.0

    k = feather_px * 2 + 1
    blend_mask = cv2.GaussianBlur(blend_mask, (k, k), 0)

    blend_3ch = blend_mask[:, :, None]

    result = (inpainted_bgr.astype(np.float32) * blend_3ch +
              original_bgr.astype(np.float32) * (1 - blend_3ch))

    return result.astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Inpainting — main function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def inpaint_background(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    method: str = "combined",
    radius: int = 7,
    dilate_iterations: int = 3,
    feather_px: int = 15,
) -> np.ndarray:
    """
    Remove the masked object and fill the hole.

    Methods:
      "basic"     – single-pass OpenCV inpainting
      "dilated"   – dilate mask first to cover edge residuals
      "multipass" – multiple inpainting passes (progressive)
      "feathered" – inpaint + smooth edge transition
      "combined"  – dilate + multipass + feather (best quality)
    """
    bool_mask = mask.astype(bool).astype(np.uint8) * 255

    if method == "basic":
        return _inpaint_basic(image_bgr, bool_mask, radius)

    elif method == "dilated":
        dilated = _dilate_mask(bool_mask, radius, dilate_iterations)
        return _inpaint_basic(image_bgr, dilated, radius)

    elif method == "multipass":
        dilated = _dilate_mask(bool_mask, radius, dilate_iterations)
        return _inpaint_multipass(image_bgr, dilated, radius, passes=3)

    elif method == "feathered":
        dilated = _dilate_mask(bool_mask, radius, dilate_iterations)
        inpainted = _inpaint_basic(image_bgr, dilated, radius)
        return _feather_edges(image_bgr, inpainted, dilated, feather_px)

    elif method == "combined":
        dilated = _dilate_mask(bool_mask, radius, dilate_iterations)
        inpainted = _inpaint_multipass(image_bgr, dilated, radius, passes=3)
        return _feather_edges(image_bgr, inpainted, dilated, feather_px)

    else:
        raise ValueError(f"Unknown method: {method}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  White background
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def white_background(image_bgr: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """Replace everything outside the mask with pure white."""
    bool_mask = mask.astype(bool)
    result = np.full_like(image_bgr, 255)
    result[bool_mask] = image_bgr[bool_mask]
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Extract object as RGBA crop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_object(image_bgr: np.ndarray,
                   mask: np.ndarray) -> Tuple[np.ndarray, BBox]:
    """Extract the masked object as a tight RGBA crop."""
    bool_mask = mask.astype(bool)
    bbox = mask_to_bbox(mask)
    crop_bgr = image_bgr[bbox.y1:bbox.y2, bbox.x1:bbox.x2].copy()
    crop_mask = bool_mask[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
    crop_rgba = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
    crop_rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)
    return crop_rgba, bbox


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Collision detection + shifting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_overlap(bbox_a: BBox, bbox_b: BBox) -> bool:
    """Check if two bounding boxes overlap."""
    if bbox_a.x1 >= bbox_b.x2 or bbox_b.x1 >= bbox_a.x2:
        return False
    if bbox_a.y1 >= bbox_b.y2 or bbox_b.y1 >= bbox_a.y2:
        return False
    return True


def compute_overlap_area(bbox_a: BBox, bbox_b: BBox) -> int:
    """Compute the overlapping area between two bboxes."""
    x_overlap = max(0, min(bbox_a.x2, bbox_b.x2) - max(bbox_a.x1, bbox_b.x1))
    y_overlap = max(0, min(bbox_a.y2, bbox_b.y2) - max(bbox_a.y1, bbox_b.y1))
    return x_overlap * y_overlap


def compute_shifted_position(
    object_bbox: BBox,
    ref_bbox: BBox,
    ref_side: str,
    gap: int = 20,
) -> int:
    """
    Compute how much to shift the object horizontally to avoid overlap.

    ref_side: "left" or "right" — where the reference object is

    Returns:
      shift_x — positive = move right, negative = move left
    """
    if not check_overlap(object_bbox, ref_bbox):
        return 0

    if ref_side == "left":
        desired_x1 = ref_bbox.x2 + gap
        shift_x = desired_x1 - object_bbox.x1
    else:
        desired_x2 = ref_bbox.x1 - gap
        shift_x = desired_x2 - object_bbox.x2

    return shift_x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Resize object and composite back
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def resize_and_composite(
    clean_bg_bgr: np.ndarray,
    object_rgba: np.ndarray,
    original_bbox: BBox,
    scale: ScaleCommand,
    padding: int = 50,
    ref_bbox: Optional[BBox] = None,
    ref_side: Optional[str] = None,
    collision_gap: int = 20,
) -> Tuple[np.ndarray, int, int, Optional[int]]:
    """
    Resize the extracted object and paste onto clean background.

    If ref_bbox is provided, checks for overlap and shifts the object
    away from the reference object.

    If the resized object overflows:
      - Canvas expands to fit (white background)
      - Canvas maintains original image aspect ratio

    Anchoring:
      - BOTTOM edge stays fixed
      - HORIZONTAL centre stays fixed

    Returns:
      (result_bgr, offset_x, offset_y, shift_x)
      shift_x = how much the object was shifted (None if no ref)
    """
    h_img, w_img = clean_bg_bgr.shape[:2]
    aspect_ratio = w_img / h_img
    obj_h, obj_w = object_rgba.shape[:2]

    # ── resize with independent factors ──
    new_w = max(1, int(obj_w * scale.w_factor))
    new_h = max(1, int(obj_h * scale.h_factor))

    max_factor = max(scale.w_factor, scale.h_factor)
    interp = cv2.INTER_AREA if max_factor < 1 else cv2.INTER_LANCZOS4
    resized = cv2.resize(object_rgba, (new_w, new_h), interpolation=interp)

    # ── compute paste position ──
    cx = (original_bbox.x1 + original_bbox.x2) / 2.0
    paste_x1 = int(cx - new_w / 2)
    paste_x2 = paste_x1 + new_w
    paste_y2 = original_bbox.y2          # bottom anchored
    paste_y1 = paste_y2 - new_h

    # ── collision detection + shift ──
    shift_x = 0
    if ref_bbox is not None and ref_side is not None:
        scaled_obj_bbox = BBox(paste_x1, paste_y1, paste_x2, paste_y2)

        shift_x = compute_shifted_position(
            scaled_obj_bbox, ref_bbox, ref_side, collision_gap
        )

        if shift_x != 0:
            paste_x1 += shift_x
            paste_x2 += shift_x

    # ── check overflow ──
    overflow_left   = max(0, -paste_x1)
    overflow_top    = max(0, -paste_y1)
    overflow_right  = max(0, paste_x2 - w_img)
    overflow_bottom = max(0, paste_y2 - h_img)

    has_overflow = (overflow_left + overflow_top +
                    overflow_right + overflow_bottom) > 0

    if not has_overflow:
        # ── simple case: fits inside image ──
        result = clean_bg_bgr.copy()
        alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
        obj_bgr = resized[:, :, :3].astype(np.float32)
        bg_region = result[paste_y1:paste_y2,
                           paste_x1:paste_x2].astype(np.float32)
        blended = obj_bgr * alpha + bg_region * (1.0 - alpha)
        result[paste_y1:paste_y2,
               paste_x1:paste_x2] = blended.astype(np.uint8)
        return result, 0, 0, shift_x if ref_bbox is not None else None

    # ── expanded canvas needed ──
    ext_left   = overflow_left   + (padding if overflow_left   > 0 else 0)
    ext_top    = overflow_top    + (padding if overflow_top    > 0 else 0)
    ext_right  = overflow_right  + (padding if overflow_right  > 0 else 0)
    ext_bottom = overflow_bottom + (padding if overflow_bottom > 0 else 0)

    req_w = w_img + ext_left + ext_right
    req_h = h_img + ext_top  + ext_bottom

    # ── maintain original aspect ratio ──
    req_ratio = req_w / req_h

    if req_ratio > aspect_ratio:
        canvas_w = req_w
        canvas_h = int(canvas_w / aspect_ratio)
    else:
        canvas_h = req_h
        canvas_w = int(canvas_h * aspect_ratio)

    extra_w = canvas_w - req_w
    extra_h = canvas_h - req_h

    ext_left   += extra_w // 2
    ext_right  += extra_w - extra_w // 2
    ext_top    += extra_h // 2
    ext_bottom += extra_h - extra_h // 2

    # ── white canvas ──
    canvas_h_final = h_img + ext_top + ext_bottom
    canvas_w_final = w_img + ext_left + ext_right
    canvas = np.full((canvas_h_final, canvas_w_final, 3), 255, dtype=np.uint8)

    img_x = ext_left
    img_y = ext_top
    canvas[img_y:img_y + h_img, img_x:img_x + w_img] = clean_bg_bgr

    # ── paste resized object ──
    canvas_paste_x1 = paste_x1 + img_x
    canvas_paste_y1 = paste_y1 + img_y
    canvas_paste_x2 = paste_x2 + img_x
    canvas_paste_y2 = paste_y2 + img_y

    clip_left   = max(0, -canvas_paste_x1)
    clip_top    = max(0, -canvas_paste_y1)
    clip_right  = max(0, canvas_paste_x2 - canvas_w_final)
    clip_bottom = max(0, canvas_paste_y2 - canvas_h_final)

    canvas_paste_x1 = max(0, canvas_paste_x1)
    canvas_paste_y1 = max(0, canvas_paste_y1)
    canvas_paste_x2 = min(canvas_w_final, canvas_paste_x2)
    canvas_paste_y2 = min(canvas_h_final, canvas_paste_y2)

    clipped_resized = resized[clip_top:new_h - clip_bottom,
                              clip_left:new_w - clip_right]

    if clipped_resized.shape[0] == 0 or clipped_resized.shape[1] == 0:
        print("  ⚠ Object scaled to zero visible size!")
        return canvas, img_x, img_y, shift_x if ref_bbox is not None else None

    alpha = clipped_resized[:, :, 3:4].astype(np.float32) / 255.0
    obj_bgr = clipped_resized[:, :, :3].astype(np.float32)
    bg_region = canvas[canvas_paste_y1:canvas_paste_y2,
                       canvas_paste_x1:canvas_paste_x2].astype(np.float32)

    blended = obj_bgr * alpha + bg_region * (1.0 - alpha)
    canvas[canvas_paste_y1:canvas_paste_y2,
           canvas_paste_x1:canvas_paste_x2] = blended.astype(np.uint8)

    return canvas, img_x, img_y, shift_x if ref_bbox is not None else None