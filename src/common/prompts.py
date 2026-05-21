"""LLM prompt definitions. Each key maps to variant list (first variant is used)."""

from __future__ import annotations

from typing import Any

PROMPTS: dict[str, list[dict[str, Any]]] = {
    "brain_decide_action": [
        {
            "prompt": (
                "Given task objective, current screenshot(s), and available tools, "
                "decide one or multiple tool calls to take to achieve the task objective.\n\n"
                "CurrentTaskGoal:\n{task}"
            ),
            "instructions": [
                "Tool calls should be in the correct order to achieve the task objective.",
                "Create detailed tool instructions for each tool call.",
                "All the monitor screenshot(s) are captured and will be provided to you.",
                "Do not do anything outside of the task scope.",
                "If task is 'click on the xxx' or '點選xxx', you should split it into move mouse to the xxx and click on the xxx.",
                "For scroll: positive clicks scroll down (往下滑), negative scroll up; use roughly 3–10 per screen of content.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "brain_decide_action_2": [
        {
            "prompt": (
                "Now you need to decide the next action to take. If the task is completed, "
                "return the reason why it is completed. Current task: {task}\n\n"
                "All monitor screenshots have been captured and will be provided to you."
            ),
            "instructions": [
                "If the previous task is not executed, try new method to achieve the task.",
                "If the tool failed to execute, do not assume the task is completed. Try new method to achieve the task.",
                "If the task can be examined by screenshot, use the screenshot to examine if the task is completed.",
                "If the task cannot be examined by screenshot, then assume the task is completed if the tool is executed successfully.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "brain_verify_script_step": [
        {
            "prompt": (
                "You are verifying whether the current scripted task step is satisfied in the screenshot. "
                "You will see the full numbered script and which step is current."
            ),
            "instructions": [
                "Decide if the current step goal is actually accomplished on screen based on visible UI and text.",
                'Return strict JSON only (no markdown), single object with keys: accomplished (bool), branch (string), target_step (number or null), reason (string).',
                "branch must be one of: advance, retry, skip, goto.",
                "Use branch advance only when accomplished is true (move to next script line).",
                "When accomplished is false: use retry to repeat the same step, skip to abandon this line and move to the next, or goto to jump to a specific script line (use target_step as the 1-based line number from the numbered list).",
                "For goto, target_step must be the line number shown before each script line (1 to N). Set target_step to null for other branches.",
                "Do not invent UI elements; base conclusions on the image and script text only.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "coordinate_selection": [
        {
            "prompt": (
                "Choose ONE line from CoordinatesText that best matches Target.\n"
                "CoordinatesText lines look like: [center_x,center_y] <OCR text for that region>.\n\n"
                "Target:\n{target}\n\n"
                "Instruction:\n{instruction}\n\n"
                "CoordinatesText:\n{coordinate_text}\n"
            ),
            "instructions": [
                "OCR text might have typos and errors, so you need to be careful to match the text correctly.",
                "Reply with the OCR text only (the part after the bracket), copied verbatim from CoordinatesText when possible so it can be matched even if it contains typos and errors.",
                "Output NOTHING except valid JSON matching the server's schema.",
                "Do not summarize, classify, bullet-list, markdown, translate, explain, add keys, or add prose.",
                "Return strict JSON only.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_element_selection": [
        {
            "prompt": (
                "Pick the candidate index from Candidates that best matches the Instruction's location hint. "
                "Each candidate row starts with [index] then center=[cx,cy] w=<width_px> h=<height_px>.\n\n"
                "Instruction:\n{instruction}\n\n"
                "Candidates:\n{candidates_text}\n"
            ),
            "instructions": [
                'Reply only with JSON: {{"index": <integer>}} — the [index] from the chosen candidate row (0-based).',
                "Never invent an index; only use an index shown in the Candidates list.",
                "No extra text or explanation.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_icon_filter": [
        {
            "prompt": (
                "Filter candidate icons by relevance to the Instruction.\n"
                "Use ONLY the attached icon crops and their [index] labels in the headers.\n\n"
                "Instruction:\n{instruction}\n\n"
                "Candidate count: {candidate_count}\n"
            ),
            "instructions": [
                'Return JSON only: {{"keep_indices": [<int>, ...]}}.',
                "keep_indices must contain only indices visible in image headers.",
                "Keep all relevant candidates; include multiple indices when uncertain.",
                "Return an empty list when no icon is relevant.",
                "Do not use or infer coordinates.",
                "Do not output prose.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_instruction_icon_location_extract": [
        {
            "prompt": (
                "Analyze the UI automation instruction below for downstream models in one response.\n\n"
                "User instruction:\n{instruction}\n"
            ),
            "instructions": [
                'Return JSON only: {{"need_text_anchor": <true|false>, "ui_icon_description": "<string>", "location_description": "<string>", "ui_shape_description": "<string>"}}.',
                "need_text_anchor: set true when the instruction refers to visible words, labels, or on-screen text content (for example: click 'Sign in', the row named X, select by caption). Set false when the target is mostly non-text visual (icon, toggle, avatar, gear, unlabeled button, panel) with no substantive text anchor.",
                "ui_icon_description: the non-text control or visual to match (icon, button, toggle, avatar, gear, etc.). No leading verbs like click/tap. Omit pure location wording.",
                "location_description: a detailed spatial description for disambiguating multiple on-screen candidates: regions (top/bottom/left/right/center, corners), relative layout (above/below/next to/beside), ordinal (first/last row), distance from window edges, header/footer/toolbar/sidebar when implied. Expand vague hints into explicit positional language. If there is no positional clue, use an empty string.",
                "ui_shape_description (optional): short visual shape, size, or aspect of the target control (e.g. small square icon, wide rounded pill, tall narrow strip, circular avatar, horizontal slider). Use an empty string when not implied or not useful for disambiguation.",
                "Do not invent UI that is not implied by the instruction.",
                "Do not output markdown or prose outside the JSON object.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
}
