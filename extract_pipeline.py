#!/usr/bin/env python3
"""
SAM2 Object Extraction Pipeline
================================

Select objects interactively and save each one as a tight‑cropped
PNG with a **transparent background**.

Workflow
────────
1. Load image
2. Draw bounding box around an object
3. SAM2 segmentation → pick the best mask
4. Crop to bounding box → save as RGBA PNG (transparent bg)
5. Repeat for more objects or quit

Usage
─────
    python extract_pipeline.py photo.jpg
    python extract_pipeline.py photo.jpg --variant tiny
    python extract_pipeline.py photo.jpg --variant large --pretrained
    python extract_pipeline.py photo.jpg -o exports/

Output
──────
    <output_dir>/object_01.png   (RGBA, transparent background)
    <output_dir>/object_02.png
    …
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from bbox_drawer import InteractivePromptDrawer, MaskEditor
from setup_model import MODEL_CONFIGS, download_checkpoint
from utils import BBox, clean_mask, extract_object, mask_to_bbox, overlay_mask


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Checkerboard helper (for preview)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _on_checkerboard(rgba: np.ndarray, block: int = 16) -> np.ndarray:
    """Composite RGBA image onto a checkerboard for visual preview."""
    rgb = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2RGB)
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    h, w = rgb.shape[:2]

    checker = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            shade = 200 if (r // block + c // block) % 2 == 0 else 255
            checker[r:r + block, c:c + block] = shade

    composite = (rgb * alpha[:, :, None]
                 + checker * (1.0 - alpha[:, :, None]))
    return composite.astype(np.uint8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ExtractPipeline:
    """Select objects with SAM2, export each as a transparent‑background PNG."""

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
            cfg = MODEL_CONFIGS[variant]
            print(f"[init] config = {cfg}")
            print(f"[init] ckpt   = {ckpt}")
            model = build_sam2(cfg, ckpt, device=self.device)
            self.predictor = SAM2ImagePredictor(model)

        self.image_bgr: Optional[np.ndarray] = None
        self.image_rgb: Optional[np.ndarray] = None

    # ── load ──────────────────────────────
    # ── load ──────────────────────────────
    def load_image(self, path: str) -> np.ndarray:
        # Use numpy to read, avoids OpenCV's special-character issue
        buf = np.fromfile(path, dtype=np.uint8)
        self.image_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if self.image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        h, w = self.image_bgr.shape[:2]
        print(f"[load] {path}  ({w}×{h})")
        return self.image_bgr

    # ── draw prompt ────────────────────────
    def select_prompt(self, object_num: int) -> Optional[Tuple[BBox, List[Tuple[int, int]], List[Tuple[int, int]]]]:
        print(f"\n┌─── Draw prompts for Object #{object_num} ────────┐")
        print("│  Step 1: Click+Drag → bbox, then SPACE to lock  │")
        print("│  Step 2: L-Click → include (+)                   │")
        print("│          R-Click → exclude (-)                   │")
        print("│          Z → undo last point                     │")
        print("│  ENTER → confirm   R → reset all   ESC → cancel │")
        print("└──────────────────────────────────────────────────┘")
        drawer = InteractivePromptDrawer()
        result = drawer.draw(self.image_bgr)
        if result:
            bbox, pos_pts, neg_pts = result
            print(f"  Selected BBox: {bbox}")
            print(f"  Positive points: {len(pos_pts)}  |  Negative points: {len(neg_pts)}")
        else:
            print("  Cancelled.")
        return result

    # ── segment ───────────────────────────
    def segment(self, bbox: BBox, pos_pts: List[Tuple[int, int]], neg_pts: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
        print("  Running SAM2 segmentation …")
        
        point_coords = None
        point_labels = None
        
        pts = pos_pts + neg_pts
        if len(pts) > 0:
            point_coords = np.array(pts, dtype=np.float32)
            point_labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int32)
            
        with torch.inference_mode():
            self.predictor.set_image(self.image_rgb)
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox.to_array()[None, :],
                multimask_output=True,
            )
        order = np.argsort(scores)[::-1]
        masks = masks[order]
        scores = scores[order]

        for i, s in enumerate(scores):
            print(f"  Mask {i}  score = {s:.4f}")
        return masks, scores

    # ── show masks & let user pick ────────
    def choose_mask(
        self, masks: np.ndarray, scores: np.ndarray, object_num: int
    ) -> Optional[int]:
        n = len(masks)
        colors = [(30, 220, 80), (220, 60, 60), (60, 60, 220)]

        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
        if n == 1:
            axes = [axes]
        for i, (mask, score) in enumerate(zip(masks, scores)):
            vis = overlay_mask(self.image_rgb, mask, colors[i % len(colors)])
            axes[i].imshow(vis)
            axes[i].set_title(f"Mask {i}  (score {score:.4f})", fontsize=12)
            axes[i].axis("off")
        plt.suptitle(
            f"Object #{object_num} — choose a mask", fontsize=14, y=1.02
        )
        plt.tight_layout()
        plt.show()

        print(f"\n  Pick mask (0‑{n - 1}), or 'n' to cancel:")
        while True:
            ans = input("  ▸ ").strip().lower()
            if ans == "n":
                return None
            try:
                idx = int(ans)
                if 0 <= idx < n:
                    print(f"  ✓ Mask {idx} selected (score {scores[idx]:.4f})")
                    return idx
            except ValueError:
                pass
            print(f"  Enter 0‑{n - 1} or 'n'")

    # ── preview extraction ────────────────
    @staticmethod
    def preview(
        image_rgb: np.ndarray,
        object_rgba: np.ndarray,
        bbox: BBox,
        mask: np.ndarray,
        object_num: int,
    ) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # panel 1: original with mask overlay
        vis = overlay_mask(image_rgb, mask, (30, 220, 80), alpha=0.45)
        axes[0].imshow(vis)
        axes[0].set_title("Mask overlay", fontsize=13)
        axes[0].axis("off")

        # panel 2: original with bbox
        axes[1].imshow(image_rgb)
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (bbox.x1, bbox.y1), bbox.width, bbox.height,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        axes[1].add_patch(rect)
        axes[1].set_title(
            f"Crop region  {bbox.width}×{bbox.height} px", fontsize=13
        )
        axes[1].axis("off")

        # panel 3: extracted object on checkerboard
        axes[2].imshow(_on_checkerboard(object_rgba))
        axes[2].set_title(
            f"Object #{object_num}  (transparent bg)\n"
            f"{object_rgba.shape[1]}×{object_rgba.shape[0]} px",
            fontsize=13,
        )
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    # ── save ──────────────────────────────
    @staticmethod
    def save(
        object_rgba: np.ndarray,
        output_dir: str,
        object_num: int,
        custom_name: Optional[str] = None,
    ) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if custom_name:
            # sanitise
            safe = "".join(
                c if c.isalnum() or c in ("_", "-") else "_"
                for c in custom_name
            )
            filename = f"{safe}.png"
        else:
            filename = f"object_{object_num:02d}.png"

        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, object_rgba)   # cv2 handles 4‑channel → RGBA PNG
        print(f"  ✓ Saved → {path}")
        return path

    # ── single‑object extraction ──────────
    def extract_one(
        self, object_num: int, output_dir: str
    ) -> Optional[str]:
        """Run the full select → segment → crop → save flow for one object."""

        # 1 – prompt
        prompt = self.select_prompt(object_num)
        if prompt is None:
            return None
        bbox, pos_pts, neg_pts = prompt

        # 2 – segment
        masks, scores = self.segment(bbox, pos_pts, neg_pts)

        # 3 – choose mask
        idx = self.choose_mask(masks, scores, object_num)
        if idx is None:
            return None
        chosen_mask = masks[idx]

        # 3.5 – refine mask (eraser / paint)
        print("\n  Opening mask editor — erase or paint to refine.")
        print("  (Press ESC to skip editing and use the mask as-is)")
        editor = MaskEditor()
        edited = editor.edit(self.image_bgr, chosen_mask)
        if edited is not None:
            chosen_mask = edited
            print("  ✓ Mask edited.")
        else:
            print("  Skipped editing — using original mask.")

        # 3.6 – clean mask (remove stray pixels / noise)
        chosen_mask = clean_mask(chosen_mask)

        # 4 – extract
        object_rgba, tight_bbox = extract_object(self.image_bgr, chosen_mask)
        print(f"  Drawn bbox:  {bbox.width}×{bbox.height} px")
        print(f"  Mask bbox:   {tight_bbox.width}×{tight_bbox.height} px  ← tight crop")
        print(f"  Crop saved:  {object_rgba.shape[1]}×{object_rgba.shape[0]} px")

        # 5 – preview
        self.preview(
            self.image_rgb, object_rgba, tight_bbox,
            chosen_mask, object_num,
        )

        # 6 – optional name
        print("\n  Name this object (or press ENTER for default):")
        name_input = input("  ▸ ").strip()

        # 7 – confirm & save
        ans = input("  Save this extraction? (y/n): ").strip().lower()
        if ans != "y":
            print("  Skipped.")
            return None

        saved = self.save(
            object_rgba, output_dir, object_num,
            custom_name=name_input if name_input else None,
        )
        return saved

    # ── main loop ─────────────────────────
    def run(self, image_path: str, output_dir: str = "extracted") -> List[str]:
        banner = """
╔══════════════════════════════════════════════════╗
║  SAM2 Object Extraction Pipeline                 ║
║  Select objects → save as transparent PNGs       ║
╚══════════════════════════════════════════════════╝"""
        print(banner)

        self.load_image(image_path)

        saved_files: List[str] = []
        object_num = 1

        while True:
            print(f"\n{'─' * 52}")
            print(f"  Object #{object_num}")
            print(f"  ({len(saved_files)} object(s) saved so far)")
            print(f"{'─' * 52}")

            result = self.extract_one(object_num, output_dir)
            if result is not None:
                saved_files.append(result)
                object_num += 1

            # continue?
            print("\n  Extract another object from the same image?")
            ans = input("  (y/n) ▸ ").strip().lower()
            if ans != "y":
                break

        # ── summary ──
        print(f"\n{'═' * 52}")
        print(f"  ✓ Done — {len(saved_files)} object(s) extracted")
        print(f"{'═' * 52}")

        if saved_files:
            print("\n  Saved files:")
            for f in saved_files:
                print(f"    • {f}")

            # final gallery
            if len(saved_files) <= 8:
                n = len(saved_files)
                cols = min(n, 4)
                rows = (n + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
                if rows == 1 and cols == 1:
                    axes = np.array([axes])
                axes = np.array(axes).flatten()

                for i, fpath in enumerate(saved_files):
                    rgba = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                    if rgba is not None:
                        axes[i].imshow(_on_checkerboard(rgba))
                        axes[i].set_title(
                            os.path.basename(fpath), fontsize=10
                        )
                    axes[i].axis("off")

                for j in range(len(saved_files), len(axes)):
                    axes[j].axis("off")

                plt.suptitle("Extracted Objects", fontsize=14)
                plt.tight_layout()
                plt.show()

        return saved_files


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(
        description="SAM2 object extraction — transparent‑background PNGs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_pipeline.py photo.jpg
  python extract_pipeline.py photo.jpg --variant tiny
  python extract_pipeline.py photo.jpg -o my_exports/
  python extract_pipeline.py photo.jpg --pretrained
        """,
    )
    ap.add_argument("image", help="Path to the input image")
    ap.add_argument(
        "-o", "--output-dir",
        default="extracted",
        help="Directory for saved PNGs (default: extracted/)",
    )
    ap.add_argument(
        "--variant",
        choices=["tiny", "small", "base", "large"],
        default="large",
        help="SAM2 model variant (default: large)",
    )
    ap.add_argument("--device", default=None, help="e.g. cuda, cpu")
    ap.add_argument(
        "--pretrained", action="store_true",
        help="Load from HuggingFace instead of local checkpoint",
    )
    args = ap.parse_args()

    pipe = ExtractPipeline(
        variant=args.variant,
        device=args.device,
        use_pretrained=args.pretrained,
    )
    pipe.run(args.image, output_dir=args.output_dir)


if __name__ == "__main__":
    main()