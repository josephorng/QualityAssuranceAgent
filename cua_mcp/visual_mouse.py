"""One-pass multimodal mouse selection over fresh YOLO/OCR candidates."""

from __future__ import annotations

from typing import Any

from cua_mcp.gemma_roi_refine import (
    metadata_from_detection,
    pick_candidate_with_gemma,
    resolve_target_via_gemma_roi,
)
from cua_mcp.screen_context import capture_screen_context
from cua_mcp.select_mouse_target import (
    _detections_similar_to,
    _maybe_disambiguate_similar_selection,
)
from src.common.run_state import get_run_state_manager, ts_name


async def resolve_visual_mouse_point(
    instruction: str,
) -> tuple[int, int, dict[str, Any]]:
    """
    Capture once, run YOLO/OCR, and ask the multimodal LLM to select one candidate.

    Unlike ``move_mouse``, this path skips target parsing, similarity filtering, and
    nearby-landmark filtering. After the one-pass pick, when other detections are
    label-similar to the chosen target, ``similar_function_describe`` re-ranks those
    peers in reading order (describe functions + re-pick).

    When YOLO/OCR returns no candidates, falls back to Gemma ROI → YOLO-on-crop →
    index pick (``selection_method: gemma_roi_yolo``).
    """
    target = (instruction or "").strip()
    if not target:
        raise ValueError("instruction must be non-empty")

    context = await capture_screen_context(include_geometry=True)
    if not context.candidates:
        grounded = await resolve_target_via_gemma_roi(
            target,
            image_paths=list(context.screenshot_paths),
            monitor_indices=list(context.monitor_indices),
        )
        if grounded is None:
            raise ValueError("No YOLO/OCR candidates found on selected monitor(s).")
        return grounded

    selected_index, selected_text, chosen = await pick_candidate_with_gemma(
        target,
        list(context.candidates),
        list(context.screenshot_paths),
    )

    disambiguation_meta: dict[str, Any] = {}
    # Re-rank only when label-similar peers exist (same path as former move_mouse).
    if len(_detections_similar_to(chosen, list(context.candidates))) > 1:
        paths = get_run_state_manager().require_paths()
        chosen, selected_index, selected_text, disambiguation_meta = (
            await _maybe_disambiguate_similar_selection(
                chosen=chosen,
                initial_idx=selected_index,
                selected_text=selected_text,
                detections=list(context.candidates),
                image_paths=list(context.screenshot_paths),
                monitor_indices=list(context.monitor_indices),
                overlay_dir=paths.yolo_ocr_dir,
                overlay_stamp=f"{ts_name()}_sim",
                anchor=target,
                nearby_phrases=[],
                nearby_matches=[],
            )
        )

    metadata = metadata_from_detection(
        chosen,
        selected_index=selected_index,
        selected_text=selected_text,
        selection_method="visual_one_pass",
        image_paths=list(context.screenshot_paths),
        instruction=target,
        extra=disambiguation_meta or None,
    )
    return chosen.cx, chosen.cy, metadata
