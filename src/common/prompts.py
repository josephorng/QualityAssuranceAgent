"""LLM prompt definitions. Each key maps to variant list (first variant is used).

``image_usage`` tiers (per prompt, at call site):
- ``no_image``: no screenshots attached; answer from text / structured lists only.
- ``optional``: images may be attached for context, but the prompt carries enough text to answer without vision.
- ``use_image``: screenshots attached and spatial or visual reasoning is required or strongly intended.
Retry prompts inherit the parent message's ``images`` list when used via ``request_json_with_retry``.
"""

from __future__ import annotations

from typing import Any, Literal

ImageUsage = Literal["no_image", "optional", "use_image"]

PROMPTS: dict[str, list[dict[str, Any]]] = {
    "brain_decide_action": [
        {
            "image_usage": "optional",
            "prompt": (
                "You are a helpful assistant that can help with tasks on the computer. "
                "Given task objective, and available tools, "
                "decide one or multiple tool calls to take to achieve the task objective.\n\n"
                "CurrentTaskGoal:\n{task}"
            ),
            "instructions": [
                "Tool calls should be in the correct order to achieve the task objective.",
                "Create detailed tool instructions for each tool call.",
                "All the monitor screenshot(s) are captured and will be provided to you.",
                "Do not do anything outside of the task scope.",
                "If task is 'click on the object' or '點選物件', you should split it into move mouse to the object and click on the object.",
                "For scroll: positive clicks scroll down (往下滑), negative scroll up; use roughly 3–10 per screen of content.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "brain_decide_action_2": [
        {
            "image_usage": "optional",
            "prompt": (
                "Now you need to decide the next action to take. If the task is completed, "
                "return the reason why it is completed. Current task: {task}\n\n"
            ),
            "instructions": [
                "If the previous task is not executed, try new method to achieve the task.",
                "If the tool failed to execute, do not assume the task is completed. Try new method to achieve the task.",
                "If the task can be examined by screenshot, use the screenshot to examine if the task is completed if the screenshot is provided.",
                "If the task cannot be examined by screenshot, then assume the task is completed if the tool is executed successfully.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "brain_verify_script_step": [
        {
            "image_usage": "use_image",
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
            "image_usage": "optional",
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
    "coordinate_disambiguation": [
        {
            "image_usage": "optional",
            "prompt": (
                "The matched OCR text appears at multiple locations in the image.\n"
                "Choose one center point (x, y) that best matches the Instruction.\n"
                "(x, y) must be one of the candidate centers listed below — "
                "same coordinate space as CoordinatesText (image pixels).\n"
                '"text" must be the OCR line for the same choice: copy from after [cx,cy] '
                "for the center you pick (verbatim when possible; OCR may have typos).\n\n"
                "Instruction:\n{instruction}\n\n"
                "Text matched in the first step:\n{chosen_text}\n\n"
                "Candidate centers (pick exactly one):\n{options_lines}\n"
            ),
            "instructions": [
                "OCR text might have typos and errors, so you need to be careful to match the text correctly.",
                "Output NOTHING except valid JSON matching the server's schema.",
                "Do not summarize, explain, or add prose.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_element_selection": [
        {
            "image_usage": "optional",
            "prompt": (
                "Pick the candidate index from Candidates that best matches the Instruction's location hint. "
                "Each candidate row starts with [index] then center=[cx,cy] w=<width_px> h=<height_px>.\n\n"
                "Instruction:\n{instruction}\n\n"
                "Candidates:\n{candidates_text}\n\n"
                "Screenshot size(s): {screenshot_sizes} (width x height pixels per monitor)."
            ),
            "instructions": [
                'Reply only with JSON: {{"index": <integer>}} — the [index] from the chosen candidate row (0-based).',
                "Never invent an index; only use an index shown in the Candidates list.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_text_filter": [
        {
            "image_usage": "no_image",
            "prompt": (
                "Select ONLY text candidates that match the user instruction.\n\n"
                "Instruction:\n{instruction}\n\n"
                "Candidates:\n{candidates_text}\n"
            ),
            "instructions": [
                'Return JSON only: {{"keep_indices": [<int>, ...]}}.',
                "Use indices from the Candidates list. Keep an empty list when none match.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "mouse_target_filter": [
        {
            "image_usage": "no_image",
            "prompt": (
                "Select ONLY candidates that are related to the user instruction.\n"
                "Each row has class=text|element|input|scrollbar, optional OCR text, and optional icon labels.\n\n"
                "Instruction:\n{instruction}\n\n"
                "Candidates:\n{candidates_text}\n"
            ),
            "instructions": [
                'Return JSON only: {{"keep_indices": [<int>, ...]}}.',
                "Use [index N] values from the Candidates list. Keep an empty list when none match.",
                "Keep text/element rows when OCR text or icons are related to the instruction.",
                "Keep input/scrollbar rows when the instruction implies a field, control, or scrollable region.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_instruction_icon_location_extract": [
        {
            "image_usage": "no_image",
            "prompt": (
                "Analyze the UI automation instruction below for downstream models in one response.\n\n"
                "User instruction:\n{instruction}\n"
            ),
            "instructions": [
                'Return JSON only: {{"need_text_anchor": <true|false>, "location_description": "<string>"}}.',
                "need_text_anchor: set true when the instruction refers to visible words, labels, or on-screen text content (for example: click 'Sign in', the row named X, select by caption). Set false when the target is mostly non-text visual (icon, toggle, avatar, gear, unlabeled button, panel) with no substantive text anchor.",
                "location_description: a detailed spatial description for disambiguating multiple on-screen candidates: regions (top/bottom/left/right/center, corners), relative layout (above/below/next to/beside), ordinal (first/last row), distance from window edges, header/footer/toolbar/sidebar when implied. Expand vague hints into explicit positional language. If there is no positional clue, use an empty string.",
                "Do not invent UI that is not implied by the instruction.",
                "Do not output markdown or prose outside the JSON object.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "hand_remap_tool": [
        {
            "image_usage": "no_image",
            "prompt": (
                "You are remapping a failed tool invocation to a valid MCP tool.\n"
                "Failed action: {action}\n"
                "Failed args JSON: {failed_args_json}\n"
                "Runtime error: {error_message}\n\n"
                "Available tools:\n{available_tools}\n\n"
                "Return JSON only with this exact shape:\n"
                '{{"action":"<one available tool name>","args":{{"...": "..."}} }}\n'
                "If args do not need changes, return the original args."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "window_select": [
        {
            "image_usage": "no_image",
            "prompt": (
                "You select one or more desktop windows to {action}.\n"
                "The user wants these windows (natural-language or partial title):\n"
                "{user_query}\n\n"
                "Additional context from the operator (use to disambiguate):\n{instruction}\n\n"
                "From the numbered list, choose every window that matches the user's intent. "
                "Use a single-element list when only one window is appropriate. "
                "Prefer main application windows over tiny dialogs or tool windows when unclear.\n"
                'Return JSON only in this exact shape: {{"indices": [<int>, ...]}}\n'
                "Use 0-based indices from the list.\n\n"
                "Windows:\n{windows_list}\n"
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    # "ui_element_selection_thinking_refine": [
    #     {
    #         "image_usage": "no_image",
    #         "prompt": (
    #             "Prior reasoning: {thinking}\n\n"
    #             "Using your prior reasoning in context and the Candidates list below, output your "
    #             'final choice as JSON only: a single object with key "index" (integer 0..{max_index}). '
    #             "No markdown, no explanation.\n\n"
    #         ),
    #         "models": ["gemma4:e2b", "gemma3:4b"],
    #     }
    # ],
    "coordinate_selection_retry": [
        {
            "image_usage": "optional",
            "prompt": (
                'Reply with ONLY: {{"text": "<string>"}} where "text" is the OCR line text from '
                "CoordinatesText (after [cx,cy] ), as verbatim as possible. "
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "coordinate_disambiguation_retry": [
        {
            "image_usage": "optional",
            "prompt": (
                'Reply with ONLY: {{"x": <integer>, "y": <integer>, "text": "<string>"}} where x,y '
                "equals one candidate [cx,cy] above and \"text\" is that line's OCR text. "
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_instruction_icon_location_extract_retry": [
        {
            "image_usage": "no_image",
            "prompt": (
                'Reply with ONLY: {{"need_text_anchor": true|false, "location_description": "..."}}. '
                "need_text_anchor: true for visible words/labels/on-screen text; false for mostly "
                "non-text targets (icon, toggle, gear, unlabeled control). location_description: "
                "detailed positional language for picking among candidates, or empty when there is "
                "no spatial clue. No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_text_filter_retry": [
        {
            "image_usage": "no_image",
            "prompt": (
                'Reply with ONLY: {{"keep_indices": [<integer>, ...]}}. '
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "mouse_target_filter_retry": [
        {
            "image_usage": "no_image",
            "prompt": (
                'Reply with ONLY: {{"keep_indices": [<integer>, ...]}}. '
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_element_selection_retry": [
        {
            "image_usage": "no_image",
            "prompt": (
                'Reply with ONLY: {{"index": <integer>}} - the [index] from the Candidates list row '
                "that best matches the location instruction (0-based). No other keys. "
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
}
