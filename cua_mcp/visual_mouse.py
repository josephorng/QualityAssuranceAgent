"""One-pass multimodal mouse selection over fresh YOLO/OCR candidates."""

from __future__ import annotations

from typing import Any

from cua_mcp.screen_context import capture_screen_context
from cua_mcp.select_mouse_target import (
    _detections_similar_to,
    _maybe_disambiguate_similar_selection,
)
from cua_mcp.select_ui_element import _parse_index_from_llm
from src.common.llm_factory import get_llm_client
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager, ts_name
from src.common.settings import load_settings


_VISUAL_MOUSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "text": {"type": "string"},
    },
    "required": ["index", "text"],
    "additionalProperties": False,
}


async def resolve_visual_mouse_point(
    instruction: str,
) -> tuple[int, int, dict[str, Any]]:
    """
    Capture once, run YOLO/OCR, and ask the multimodal LLM to select one candidate.

    Unlike ``move_mouse``, this path skips target parsing, similarity filtering, and
    nearby-landmark filtering. After the one-pass pick, when other detections are
    label-similar to the chosen target, ``similar_function_describe`` re-ranks those
    peers (describe functions + re-pick).
    """
    target = (instruction or "").strip()
    if not target:
        raise ValueError("instruction must be non-empty")

    context = await capture_screen_context(include_geometry=True)
    if not context.candidates:
        raise ValueError("No YOLO/OCR candidates found on selected monitor(s).")

    prompt = get_prompt("visual_mouse_selection").format(
        instruction=target,
        candidates_text=context.ocr_text,
    )
    response = await get_llm_client().chat_messages(
        load_settings().brain_lm,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": list(context.screenshot_paths),
            }
        ],
        tools=[],
        response_format=_VISUAL_MOUSE_SCHEMA,
        think=True,
    )
    selected_index, selected_text = _parse_index_from_llm(
        response.content,
        len(context.candidates),
    )
    chosen = context.candidates[selected_index]

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

    x, y, w, h = chosen.bbox
    metadata: dict[str, Any] = {
        "selected_index": selected_index,
        "selected_text": selected_text,
        "selection_method": "visual_one_pass",
        "screenshot_path": context.screenshot_paths[0]
        if context.screenshot_paths
        else "",
        "screenshot_paths": list(context.screenshot_paths),
        "target_kind": chosen.class_name,
        "target_text": chosen.text or "",
        "target_icons": list(chosen.icons or []),
        "target_bbox": {"x": x, "y": y, "w": w, "h": h},
        "image_center": {"x": chosen.cx, "y": chosen.cy},
        "resolved_center": {"x": chosen.cx, "y": chosen.cy},
        "anchor_instruction": target,
    }
    if disambiguation_meta:
        metadata.update(disambiguation_meta)
    return chosen.cx, chosen.cy, metadata
