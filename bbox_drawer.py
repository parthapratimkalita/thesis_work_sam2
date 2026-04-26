"""
Interactive bounding‑box drawer using OpenCV high‑GUI.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import BBox


class BoundingBoxDrawer:
    """Opens an OpenCV window and lets the user draw a rectangle."""

    def __init__(self) -> None:
        self._drawing = False
        self._ix = self._iy = -1
        self._fx = self._fy = -1
        self._bbox: Optional[BBox] = None

    # ── mouse callback ──────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._ix, self._iy = x, y
            self._fx, self._fy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._fx, self._fy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._fx, self._fy = x, y
            x1, y1 = min(self._ix, x), min(self._iy, y)
            x2, y2 = max(self._ix, x), max(self._iy, y)
            if x2 - x1 > 2 and y2 - y1 > 2:          # ignore tiny clicks
                self._bbox = BBox(x1, y1, x2, y2)

    # ── public API ──────────────────────────
    def draw(self, image: np.ndarray) -> Optional[BBox]:
        """
        Show *image* in a window; return the drawn BBox or None.

        Controls
        ────────
        Click + Drag   → draw rectangle
        ENTER          → confirm
        R              → reset
        ESC            → cancel
        """
        win = "Draw Bounding Box  |  ENTER=confirm  R=reset  ESC=cancel"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(image.shape[1], 1280), min(image.shape[0], 900))
        cv2.setMouseCallback(win, self._on_mouse)

        clone = image.copy()

        while True:
            frame = clone.copy()

            # live rubber‑band
            if self._drawing and self._ix >= 0:
                cv2.rectangle(frame,
                              (self._ix, self._iy),
                              (self._fx, self._fy),
                              (0, 255, 0), 2)
            elif self._bbox is not None:
                cv2.rectangle(frame,
                              (self._bbox.x1, self._bbox.y1),
                              (self._bbox.x2, self._bbox.y2),
                              (0, 255, 0), 2)
                # label
                cv2.putText(frame,
                            f"{self._bbox.width}x{self._bbox.height}",
                            (self._bbox.x1, self._bbox.y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == 13 and self._bbox is not None:   # ENTER
                cv2.destroyWindow(win)
                return self._bbox
            if key == ord("r"):                         # reset
                self._bbox = None
            if key == 27:                               # ESC
                cv2.destroyWindow(win)
                return None

        return None                                     # unreachable


class InteractivePromptDrawer:
    """
    Two-stage interactive prompt drawer:
      Stage 1 — draw a bounding box (click + drag)
      Stage 2 — add positive / negative point prompts (click)

    Returns:
        (BBox, list of pos_points, list of neg_points)
        where points are (x, y) tuples in image coordinates.
    """

    # Visual settings – scaled relative to image size later
    _PT_RADIUS = 8
    _PT_OUTLINE = 2
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self) -> None:
        self._mode = "BBOX"          # "BBOX" or "POINTS"
        self._drawing = False
        self._ix = self._iy = -1
        self._fx = self._fy = -1
        self._bbox: Optional[BBox] = None

        self._pos_points: List[Tuple[int, int]] = []
        self._neg_points: List[Tuple[int, int]] = []

        # Ordered list so we can undo any point regardless of type
        self._point_history: List[Tuple[str, Tuple[int, int]]] = []

    # ── mouse callback ──────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if self._mode == "BBOX":
            if event == cv2.EVENT_LBUTTONDOWN:
                self._drawing = True
                self._ix, self._iy = x, y
                self._fx, self._fy = x, y

            elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
                self._fx, self._fy = x, y

            elif event == cv2.EVENT_LBUTTONUP:
                self._drawing = False
                self._fx, self._fy = x, y
                x1, y1 = min(self._ix, x), min(self._iy, y)
                x2, y2 = max(self._ix, x), max(self._iy, y)
                if x2 - x1 > 2 and y2 - y1 > 2:
                    self._bbox = BBox(x1, y1, x2, y2)

        elif self._mode == "POINTS":
            if event == cv2.EVENT_LBUTTONDOWN:
                self._pos_points.append((x, y))
                self._point_history.append(("pos", (x, y)))
            elif event == cv2.EVENT_RBUTTONDOWN:
                self._neg_points.append((x, y))
                self._point_history.append(("neg", (x, y)))

    # ── HUD overlay ─────────────────────────
    @staticmethod
    def _draw_hud(frame: np.ndarray, lines: List[str],
                  *, origin: Tuple[int, int] = (10, 30),
                  scale: float = 0.55, color=(255, 255, 255),
                  bg_color=(0, 0, 0), thickness: int = 1) -> None:
        """Draw semi-transparent text overlay in the top-left corner."""
        x0, y0 = origin
        line_h = int(25 * scale / 0.55)

        # compute background rect
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                          scale, thickness)
            max_w = max(max_w, tw)

        pad = 8
        rect_h = line_h * len(lines) + pad * 2
        rect_w = max_w + pad * 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0 - pad, y0 - line_h - pad + 4),
                      (x0 + rect_w, y0 + line_h * (len(lines) - 1) + pad),
                      bg_color, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, line in enumerate(lines):
            cv2.putText(frame, line,
                        (x0, y0 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                        cv2.LINE_AA)

    # ── draw points on frame ────────────────
    def _draw_points(self, frame: np.ndarray) -> None:
        r = self._PT_RADIUS
        for pt in self._pos_points:
            cv2.circle(frame, pt, r, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, r, (255, 255, 255), self._PT_OUTLINE, cv2.LINE_AA)
            # draw "+" inside
            cv2.line(frame, (pt[0] - r // 2, pt[1]), (pt[0] + r // 2, pt[1]),
                     (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (pt[0], pt[1] - r // 2), (pt[0], pt[1] + r // 2),
                     (255, 255, 255), 1, cv2.LINE_AA)

        for pt in self._neg_points:
            cv2.circle(frame, pt, r, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, r, (255, 255, 255), self._PT_OUTLINE, cv2.LINE_AA)
            # draw "-" inside
            cv2.line(frame, (pt[0] - r // 2, pt[1]), (pt[0] + r // 2, pt[1]),
                     (255, 255, 255), 1, cv2.LINE_AA)

    # ── undo last point ─────────────────────
    def _undo_last_point(self) -> None:
        if not self._point_history:
            return
        kind, pt = self._point_history.pop()
        if kind == "pos" and pt in self._pos_points:
            self._pos_points.remove(pt)
        elif kind == "neg" and pt in self._neg_points:
            self._neg_points.remove(pt)

    # ── public API ──────────────────────────
    def draw(
        self, image: np.ndarray
    ) -> Optional[Tuple[BBox, List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """
        Show *image* in a window; return (BBox, pos_points, neg_points) or None.

        Controls – BBox mode
        ─────────────────────
        Click + Drag   → draw rectangle
        SPACE          → lock box, switch to Point mode
        R              → redraw box
        ESC            → cancel

        Controls – Point mode
        ──────────────────────
        Left-click     → positive point  (include)
        Right-click    → negative point  (exclude)
        Z              → undo last point
        ENTER          → confirm & run SAM2
        R              → reset everything
        ESC            → cancel
        """
        win = "SAM2 Interactive Prompt"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(image.shape[1], 1280),
                         min(image.shape[0], 900))
        cv2.setMouseCallback(win, self._on_mouse)

        clone = image.copy()

        while True:
            frame = clone.copy()

            # ── draw bbox ──
            if self._mode == "BBOX" and self._drawing and self._ix >= 0:
                cv2.rectangle(frame,
                              (self._ix, self._iy),
                              (self._fx, self._fy),
                              (0, 255, 0), 2)
            elif self._bbox is not None:
                cv2.rectangle(frame,
                              (self._bbox.x1, self._bbox.y1),
                              (self._bbox.x2, self._bbox.y2),
                              (0, 255, 0), 2)
                if self._mode == "BBOX":
                    cv2.putText(frame,
                                f"{self._bbox.width}x{self._bbox.height}",
                                (self._bbox.x1, self._bbox.y1 - 8),
                                self._FONT, 0.6, (0, 255, 0), 2)

            # ── draw points ──
            self._draw_points(frame)

            # ── HUD ──
            if self._mode == "BBOX":
                hud = [
                    "STEP 1: Draw Bounding Box",
                    "Click+Drag = box | SPACE = lock | R = redo | ESC = quit",
                ]
            else:
                n_pos = len(self._pos_points)
                n_neg = len(self._neg_points)
                hud = [
                    "STEP 2: Add Point Prompts  (optional)",
                    f"L-Click = +include ({n_pos})  |  R-Click = -exclude ({n_neg})",
                    "Z = undo  |  ENTER = confirm  |  R = reset all  |  ESC = quit",
                ]
            self._draw_hud(frame, hud)

            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF

            # ── global keys ──
            if key == 27:                              # ESC
                cv2.destroyWindow(win)
                return None

            if key == ord("r") or key == ord("R"):     # Reset
                self._bbox = None
                self._pos_points.clear()
                self._neg_points.clear()
                self._point_history.clear()
                self._mode = "BBOX"
                continue

            # ── mode-specific keys ──
            if self._mode == "BBOX":
                if key == 32 and self._bbox is not None:          # SPACE only
                    self._mode = "POINTS"

            elif self._mode == "POINTS":
                if key == ord("z") or key == ord("Z"):            # Undo
                    self._undo_last_point()
                elif key == 13:                                   # ENTER
                    cv2.destroyWindow(win)
                    return (self._bbox,
                            list(self._pos_points),
                            list(self._neg_points))

        return None  # unreachable


class MaskEditor:
    """
    Interactive mask editor — erase or paint regions of a binary mask.

    Controls
    ────────
    Left-click drag    → erase  (remove from mask)
    Right-click drag   → paint  (add to mask)
    [ / scroll-down    → shrink brush
    ] / scroll-up      → grow brush
    Z                  → undo last stroke
    ENTER              → confirm edited mask
    R                  → reset to original mask
    ESC                → cancel (keep original mask)
    """

    _MAX_UNDO = 50
    _MIN_BRUSH = 3
    _MAX_BRUSH = 120
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, brush_radius: int = 20) -> None:
        self._brush_radius = brush_radius
        self._painting = False
        self._erase_mode = True
        self._mx = self._my = -1     # live cursor position
        self._mask: Optional[np.ndarray] = None
        self._original_mask: Optional[np.ndarray] = None
        self._undo_stack: List[np.ndarray] = []

    # ── mouse callback ──────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        self._mx, self._my = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            self._painting = True
            self._erase_mode = True
            self._push_undo()
            self._apply_brush(x, y, erase=True)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self._painting = True
            self._erase_mode = False
            self._push_undo()
            self._apply_brush(x, y, erase=False)

        elif event == cv2.EVENT_MOUSEMOVE and self._painting:
            self._apply_brush(x, y, erase=self._erase_mode)

        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._painting = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self._brush_radius = min(self._MAX_BRUSH,
                                         self._brush_radius + 2)
            else:
                self._brush_radius = max(self._MIN_BRUSH,
                                         self._brush_radius - 2)

    # ── helpers ─────────────────────────────
    def _push_undo(self) -> None:
        if self._mask is None:
            return
        if len(self._undo_stack) >= self._MAX_UNDO:
            self._undo_stack.pop(0)
        self._undo_stack.append(self._mask.copy())

    def _pop_undo(self) -> None:
        if self._undo_stack:
            self._mask = self._undo_stack.pop()

    def _apply_brush(self, x: int, y: int, erase: bool) -> None:
        if self._mask is None:
            return
        value = 0 if erase else 1
        cv2.circle(self._mask, (x, y), self._brush_radius,
                   int(value), -1)

    # ── HUD (reuse pattern from InteractivePromptDrawer) ──
    @staticmethod
    def _draw_hud(frame: np.ndarray, lines: List[str],
                  *, origin: Tuple[int, int] = (10, 30),
                  scale: float = 0.50, color=(255, 255, 255),
                  bg_color=(0, 0, 0), thickness: int = 1) -> None:
        x0, y0 = origin
        line_h = int(25 * scale / 0.55)
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX,
                                          scale, thickness)
            max_w = max(max_w, tw)
        pad = 8
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x0 - pad, y0 - line_h - pad + 4),
                      (x0 + max_w + pad * 2,
                       y0 + line_h * (len(lines) - 1) + pad),
                      bg_color, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x0, y0 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                        cv2.LINE_AA)

    # ── public API ──────────────────────────
    def edit(self, image_bgr: np.ndarray,
             mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Show the image with mask overlay and let the user erase/paint.

        Returns the edited boolean mask (H, W) or None if cancelled.
        If cancelled, the caller should use the original mask.
        """
        self._mask = mask.astype(np.uint8).copy()
        self._original_mask = mask.astype(np.uint8).copy()
        self._undo_stack.clear()

        win = "SAM2 Mask Editor"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(image_bgr.shape[1], 1280),
                         min(image_bgr.shape[0], 900))
        cv2.setMouseCallback(win, self._on_mouse)

        while True:
            # ── composite: image + green mask overlay ──
            frame = image_bgr.copy().astype(np.float32)
            bool_mask = self._mask.astype(bool)
            overlay_color = np.array([0, 200, 60], dtype=np.float32)
            frame[bool_mask] = (frame[bool_mask] * 0.55
                                + overlay_color * 0.45)

            # dim the erased area slightly so it's obvious what's gone
            frame[~bool_mask] = frame[~bool_mask] * 0.70

            frame = frame.astype(np.uint8)

            # ── brush cursor ──
            if self._mx >= 0 and self._my >= 0:
                # white circle outline for brush
                cv2.circle(frame, (self._mx, self._my),
                           self._brush_radius, (255, 255, 255), 1,
                           cv2.LINE_AA)
                # small crosshair
                cv2.line(frame,
                         (self._mx - 4, self._my),
                         (self._mx + 4, self._my),
                         (255, 255, 255), 1, cv2.LINE_AA)
                cv2.line(frame,
                         (self._mx, self._my - 4),
                         (self._mx, self._my + 4),
                         (255, 255, 255), 1, cv2.LINE_AA)

            # ── HUD ──
            undo_n = len(self._undo_stack)
            hud = [
                "MASK EDITOR  —  Refine before saving",
                f"L-Drag = erase  |  R-Drag = paint back  |  Brush: {self._brush_radius}px",
                f"[ ] or Scroll = brush size  |  Z = undo ({undo_n})",
                "ENTER = confirm  |  R = reset  |  ESC = skip editing",
            ]
            self._draw_hud(frame, hud)

            cv2.imshow(win, frame)
            key = cv2.waitKey(20) & 0xFF

            # ── keys ──
            if key == 27:                              # ESC — skip
                cv2.destroyWindow(win)
                return None

            if key == 13:                              # ENTER — confirm
                cv2.destroyWindow(win)
                return self._mask.astype(bool)

            if key == ord("r") or key == ord("R"):     # Reset
                self._mask = self._original_mask.copy()
                self._undo_stack.clear()

            if key == ord("z") or key == ord("Z"):     # Undo
                self._pop_undo()

            if key == ord("["):                         # Shrink brush
                self._brush_radius = max(self._MIN_BRUSH,
                                         self._brush_radius - 2)
            if key == ord("]"):                         # Grow brush
                self._brush_radius = min(self._MAX_BRUSH,
                                         self._brush_radius + 2)

        return None  # unreachable