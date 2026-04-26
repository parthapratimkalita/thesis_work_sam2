#!/usr/bin/env python3
"""
SAM2 Interactive Segmentation Pipeline
=======================================

Workflow
────────
1. Load image
2. Draw bounding box → select object
3. SAM2 segmentation → pick a mask
4. Choose background method
5. Optional: mark reference object for collision avoidance
6. Scale the object interactively
7. Save results
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from bbox_drawer import BoundingBoxDrawer
from setup_model import MODEL_CONFIGS, download_checkpoint
from utils import (
    BBox,
    ScaleCommand,
    extract_object,
    inpaint_background,
    white_background,
    mask_to_bbox,
    overlay_mask,
    parse_scale_input,
    resize_and_composite,
)


class SAM2Pipeline:
    """End‑to‑end interactive segmentation + object scaling pipeline."""

    def __init__(
        self,
        variant: str = "large",
        device: Optional[str] = None,
        use_pretrained: bool = False,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[init] device = {self.device}")

        if use_pretrained:
            hf_name = f"facebook/sam2.1-hiera-{variant}"
            print(f"[init] Loading pretrained model: {hf_name}")
            self.predictor = SAM2ImagePredictor.from_pretrained(hf_name)
        else:
            ckpt = download_checkpoint(variant)
            cfg  = MODEL_CONFIGS[variant]
            print(f"[init] config   = {cfg}")
            print(f"[init] ckpt     = {ckpt}")
            model = build_sam2(cfg, ckpt, device=self.device)
            self.predictor = SAM2ImagePredictor(model)

        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None
        self.masks:  Optional[np.ndarray] = None
        self.scores: Optional[np.ndarray] = None

    # ── 1. load image ─────────────────────
    def load_image(self, path: str) -> np.ndarray:
        self.image_bgr = cv2.imread(path)
        if self.image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        h, w = self.image_bgr.shape[:2]
        print(f"[load] {path}  ({w}×{h})")
        return self.image_bgr

    # ── 2. interactive bbox ────────────────
    def select_bbox(self) -> Optional[BBox]:
        print("\n┌─── STEP 1: Draw a bounding box ───┐")
        drawer = BoundingBoxDrawer()
        bbox = drawer.draw(self.image_bgr)
        if bbox:
            print(f"│  Selected {bbox}")
        else:
            print("│  Cancelled.")
        print("└────────────────────────────────────┘")
        return bbox

    # ── 3. SAM2 segmentation ──────────────
    def segment(self, bbox: BBox) -> Tuple[np.ndarray, np.ndarray]:
        print("\n┌─── STEP 2: SAM2 segmentation ─────┐")
        with torch.inference_mode():
            self.predictor.set_image(self.image_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox.to_array()[None, :],
                multimask_output=True,
            )

        order = np.argsort(scores)[::-1]
        self.masks  = masks[order]
        self.scores = scores[order]

        for i, s in enumerate(self.scores):
            print(f"│  Mask {i}  score={s:.4f}")
        print("└────────────────────────────────────┘")
        return self.masks, self.scores

    # ── 4. display & confirm mask ─────────
    def show_masks(self) -> None:
        n = len(self.masks)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]
        colors = [(30, 220, 80), (220, 60, 60), (60, 60, 220)]
        for i, (mask, score) in enumerate(zip(self.masks, self.scores)):
            vis = overlay_mask(self.image_rgb, mask, colors[i % len(colors)])
            axes[i].imshow(vis)
            axes[i].set_title(f"Mask {i}  (score {score:.4f})", fontsize=13)
            axes[i].axis("off")
        plt.suptitle("Choose a mask index below", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()

    def confirm_mask(self) -> Optional[int]:
        self.show_masks()
        print("\n┌─── STEP 3: Confirm a mask ────────┐")
        print(f"│  Enter 0‑{len(self.masks)-1} to pick, 'n' to cancel")
        while True:
            ans = input("│  ▸ ").strip().lower()
            if ans == "n":
                print("└── Cancelled. ─────────────────────┘")
                return None
            try:
                idx = int(ans)
                if 0 <= idx < len(self.masks):
                    print(f"│  ✓ Mask {idx} confirmed.")
                    print("└────────────────────────────────────┘")
                    return idx
            except ValueError:
                pass
            print(f"│  Invalid — enter 0‑{len(self.masks)-1} or 'n'")

    # ── 5. choose background method ───────
    def choose_background(self, chosen_mask: np.ndarray) -> np.ndarray:
        print("\n┌─── STEP 4: Choose background method ──────────┐")
        print("│                                                │")
        print("│  1 = Combined (best quality inpainting)        │")
        print("│  2 = Basic inpainting (fast, may have residual)│")
        print("│  3 = Multipass inpainting (better, slower)     │")
        print("│  4 = Feathered inpainting (smooth edges)       │")
        print("│  5 = Pure white background (guaranteed clean)  │")
        print("│                                                │")
        print("└────────────────────────────────────────────────┘")

        methods = {
            "1": "combined",
            "2": "basic",
            "3": "multipass",
            "4": "feathered",
            "5": "white",
        }

        while True:
            bg_choice = input("  ▸ ").strip()
            if bg_choice in methods:
                break
            print("  Enter 1-5")

        if bg_choice == "5":
            print("  Using white background …")
            clean_bg = white_background(self.image_bgr, chosen_mask)
        else:
            method = methods[bg_choice]
            print(f"  Using {method} inpainting …")
            clean_bg = inpaint_background(
                self.image_bgr, chosen_mask, method=method
            )

        return clean_bg

    # ── 6. optional reference object ──────
    def select_reference_object(self) -> Tuple[Optional[BBox], Optional[str]]:
        """Let user optionally mark a neighboring object to avoid."""
        print("\n┌─── STEP 5: Collision avoidance (optional) ────┐")
        print("│                                                │")
        print("│  Is there a neighboring object you want to     │")
        print("│  avoid overlapping when scaling?                │")
        print("│                                                │")
        print("│  y = yes, draw reference object bbox           │")
        print("│  n = no, skip collision detection              │")
        print("│                                                │")
        print("└────────────────────────────────────────────────┘")

        ans = input("  ▸ ").strip().lower()
        if ans != "y":
            print("  Skipping collision detection.")
            return None, None

        # Draw reference bbox
        print("\n  Draw a bounding box around the REFERENCE object.")
        print("  (The object you want to AVOID overlapping with)")
        drawer = BoundingBoxDrawer()
        ref_bbox = drawer.draw(self.image_bgr)

        if ref_bbox is None:
            print("  Cancelled. Skipping collision detection.")
            return None, None

        print(f"  Reference object: {ref_bbox}")

        # Ask which side
        print("\n  Where is this reference object relative to YOUR object?")
        print("    l = left")
        print("    r = right")

        while True:
            side = input("  ▸ ").strip().lower()
            if side in ("l", "left"):
                ref_side = "left"
                break
            elif side in ("r", "right"):
                ref_side = "right"
                break
            print("  Enter 'l' for left or 'r' for right")

        print(f"  ✓ Reference object on the {ref_side}")

        # Show reference bbox on image
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB))
        self._draw_rect(ax, ref_bbox, "orange",
                        f"reference ({ref_side})", ls="--")
        ax.set_title("Reference object marked", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        return ref_bbox, ref_side

    # ── 7. drawing helper ─────────────────
    @staticmethod
    def _draw_rect(ax, bbox: BBox, color: str, label: str, ls: str = "-"):
        rect = mpatches.FancyBboxPatch(
            (bbox.x1, bbox.y1), bbox.width, bbox.height,
            linewidth=2, edgecolor=color, facecolor="none",
            linestyle=ls, boxstyle="square,pad=0",
        )
        ax.add_patch(rect)
        ax.annotate(label,
                    (bbox.x1, bbox.y1 - 6),
                    color=color, fontsize=10, fontweight="bold")

    # ── 8. show result with bboxes ────────
    def show_result(
        self,
        original_rgb: np.ndarray,
        result_rgb: np.ndarray,
        original_bbox: BBox,
        scale: ScaleCommand,
        offset_x: int = 0,
        offset_y: int = 0,
        ref_bbox: Optional[BBox] = None,
        ref_side: Optional[str] = None,
        shift_x: Optional[int] = None,
    ) -> None:
        """Show original vs scaled side by side."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # ── left panel: original ──
        ax1.imshow(original_rgb)
        self._draw_rect(ax1, original_bbox, "cyan",
                        f"original {original_bbox.width}×{original_bbox.height}")
        if ref_bbox is not None:
            self._draw_rect(ax1, ref_bbox, "orange",
                            f"reference ({ref_side})", ls="--")
        ax1.set_title("Original", fontsize=14)
        ax1.axis("off")

        # ── right panel: result ──
        ax2.imshow(result_rgb)

        # Bottom-anchored scaled bbox — with shift and offset
        cx = (original_bbox.x1 + original_bbox.x2) / 2.0
        new_w = int(original_bbox.width * scale.w_factor)
        new_h = int(original_bbox.height * scale.h_factor)

        actual_shift = shift_x if shift_x is not None else 0
        sx1 = int(cx - new_w / 2) + offset_x + actual_shift
        sy2 = original_bbox.y2 + offset_y
        sy1 = sy2 - new_h
        sx2 = sx1 + new_w

        h_canvas, w_canvas = result_rgb.shape[:2]
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(w_canvas, sx2), min(h_canvas, sy2)

        scaled_bbox = BBox(sx1, sy1, sx2, sy2)
        label = f"scaled {scaled_bbox.width}×{scaled_bbox.height}"
        self._draw_rect(ax2, scaled_bbox, "red", label)

        # Show reference on result too
        if ref_bbox is not None:
            ref_shifted = BBox(
                ref_bbox.x1 + offset_x, ref_bbox.y1 + offset_y,
                ref_bbox.x2 + offset_x, ref_bbox.y2 + offset_y,
            )
            self._draw_rect(ax2, ref_shifted, "orange",
                            f"reference ({ref_side})", ls="--")

        # Show original image boundary if canvas expanded
        if offset_x > 0 or offset_y > 0:
            h_orig, w_orig = original_rgb.shape[:2]
            orig_on_canvas = BBox(offset_x, offset_y,
                                  offset_x + w_orig, offset_y + h_orig)
            self._draw_rect(ax2, orig_on_canvas, "gray",
                            "original image boundary", ls="--")

        # Title
        title = f"Resized: {scale}"
        if shift_x is not None and shift_x != 0:
            direction = "right" if shift_x > 0 else "left"
            title += f"  |  shifted {abs(shift_x)}px {direction}"
        ax2.set_title(title, fontsize=14)
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    # ── 9. interactive object scaling loop ─
    def interactive_scaling(
        self,
        mask: np.ndarray,
        clean_bg_bgr: np.ndarray,
        object_rgba: np.ndarray,
        original_bbox: BBox,
        ref_bbox: Optional[BBox] = None,
        ref_side: Optional[str] = None,
    ) -> None:
        print("\n┌─── STEP 6: Resize the object ──────────────────────┐")
        print("│                                                     │")
        print("│  Object resizes. Background stays unchanged.        │")
        print("│  Bottom edge stays anchored (grounded).             │")
        print("│  If object overflows → canvas expands (white bg).   │")
        if ref_bbox is not None:
            print("│  ⚡ Collision detection ACTIVE                     │")
            print(f"│     Reference object on the {ref_side:5s}                │")
        print("│                                                     │")
        print("│  NATURAL LANGUAGE (additive):                       │")
        print("│    '1x bigger'    → double    (factor 2.0)          │")
        print("│    '2x bigger'    → triple    (factor 3.0)          │")
        print("│    '1x smaller'   → half      (factor 0.5)         │")
        print("│    '2x smaller'   → third     (factor 0.33)        │")
        print("│    '1x wider'     → width doubles, height same     │")
        print("│    '2x taller'    → height triples, width same     │")
        print("│    '1x narrower'  → width halves, height same      │")
        print("│    '2x shorter'   → height shrinks ⅓, width same   │")
        print("│                                                     │")
        print("│  DIRECT MULTIPLIER:                                 │")
        print("│    '2'            → factor 2.0 (double)             │")
        print("│    '0.5'          → factor 0.5 (half)               │")
        print("│    'w2 h3'        → width ×2, height ×3             │")
        print("│                                                     │")
        print("│  'q' → quit                                        │")
        print("└─────────────────────────────────────────────────────┘")

        original_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)

        while True:
            raw = input("\n  Scale ▸ ").strip()
            if raw.lower() == "q":
                break

            scale = parse_scale_input(raw)
            if scale is None:
                print("  ✗ Could not parse. See commands above.")
                continue

            # ── resize + collision detection ──
            result_bgr, offset_x, offset_y, shift_x = resize_and_composite(
                clean_bg_bgr, object_rgba, original_bbox, scale,
                ref_bbox=ref_bbox,
                ref_side=ref_side,
            )
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            if offset_x > 0 or offset_y > 0:
                print("  ⓘ Canvas expanded (object overflowed image bounds)")

            if shift_x is not None and shift_x != 0:
                direction = "right" if shift_x > 0 else "left"
                print(f"  ⚡ Collision detected! Object shifted "
                      f"{abs(shift_x)}px {direction}")

            print(f"  {scale}")
            self.show_result(
                original_rgb, result_rgb, original_bbox, scale,
                offset_x, offset_y,
                ref_bbox, ref_side, shift_x,
            )

            ans = input("  Save this result? (y / n): ").strip().lower()
            if ans == "y":
                tag = f"w{scale.w_factor:.2f}_h{scale.h_factor:.2f}"
                out = f"result_{tag}.png"
                cv2.imwrite(out, result_bgr)
                print(f"  ✓ Saved → {out}")

    # ── 10. full run ──────────────────────
    def run(self, image_path: str) -> Optional[Dict]:
        banner = """
╔══════════════════════════════════════════════╗
║  SAM2 Interactive Object Resizing Pipeline   ║
╚══════════════════════════════════════════════╝"""
        print(banner)

        # 1 – load
        self.load_image(image_path)

        # 2 – draw bbox
        bbox = self.select_bbox()
        if bbox is None:
            return None

        # 3 – segment
        self.segment(bbox)

        # 4 – confirm mask
        idx = self.confirm_mask()
        if idx is None:
            return None
        chosen_mask = self.masks[idx]

        # 5 – extract object + choose background
        print("\n  Extracting object …")
        object_rgba, tight_bbox = extract_object(self.image_bgr, chosen_mask)
        print(f"  Object bbox: {tight_bbox}")
        print(f"  Object size: {object_rgba.shape[1]}×{object_rgba.shape[0]} px")

        clean_bg = self.choose_background(chosen_mask)

        # Show extraction result
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        ax1.imshow(cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original", fontsize=13)
        ax1.axis("off")

        ax2.imshow(cv2.cvtColor(clean_bg, cv2.COLOR_BGR2RGB))
        ax2.set_title("Background (cleaned)", fontsize=13)
        ax2.axis("off")

        # Extracted object on checkerboard
        obj_display = object_rgba.copy()
        obj_rgb = cv2.cvtColor(obj_display[:, :, :3], cv2.COLOR_BGR2RGB)
        obj_alpha = obj_display[:, :, 3].astype(float) / 255.0

        ch, cw = obj_rgb.shape[:2]
        checker = np.zeros((ch, cw, 3), dtype=np.uint8)
        block = 16
        for r in range(0, ch, block):
            for c in range(0, cw, block):
                if (r // block + c // block) % 2 == 0:
                    checker[r:r + block, c:c + block] = [200, 200, 200]
                else:
                    checker[r:r + block, c:c + block] = [255, 255, 255]

        obj_on_checker = (
            obj_rgb * obj_alpha[:, :, None]
            + checker * (1 - obj_alpha[:, :, None])
        ).astype(np.uint8)
        ax3.imshow(obj_on_checker)
        ax3.set_title("Extracted Object", fontsize=13)
        ax3.axis("off")

        plt.tight_layout()
        plt.show()

        # 6 – optional reference object for collision detection
        ref_bbox, ref_side = self.select_reference_object()

        # 7 – interactive scaling
        self.interactive_scaling(
            chosen_mask, clean_bg, object_rgba, tight_bbox,
            ref_bbox, ref_side,
        )

        print("\n✓ Pipeline complete.")
        return {
            "mask": chosen_mask,
            "score": float(self.scores[idx]),
            "bbox": tight_bbox,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(
        description="SAM2 interactive object resizing pipeline")
    ap.add_argument("image", help="Path to the input image")
    ap.add_argument("--variant",
                    choices=["tiny", "small", "base", "large"],
                    default="large")
    ap.add_argument("--device", default=None)
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    pipe = SAM2Pipeline(
        variant=args.variant,
        device=args.device,
        use_pretrained=args.pretrained,
    )
    pipe.run(args.image)


if __name__ == "__main__":
    main()