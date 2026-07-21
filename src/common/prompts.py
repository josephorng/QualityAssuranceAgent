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
                "Translate CurrentTaskGoal into one or more tool calls using the available tools. "
                "Map the goal literally—do not invent preparatory steps, targets, or visibility "
                "checks that are not stated in the goal.\n\n"
                "CurrentTaskGoal:\n{task}"
            ),
            "instructions": [
                "Tool calls should be in the correct order to achieve the task objective.",
                "Create detailed tool instructions for each tool call.",
                "All the monitor screenshot(s) are captured and will be provided to you.",
                "Do not do anything outside of the task scope.",
                "Do not add click/move_mouse/check_object_exists unless the goal names a target or explicitly asks to check visibility (如果畫面上有… / 如果畫面上沒有… / if … is visible / if … is not visible).",
                "For steps that depend on whether something is on screen (如果畫面上有… / 如果畫面上沒有… / if … is visible / if … is not visible): call check_object_exists first with the target in instruction (and nearby_objects when given). Do not call move_mouse, click, or other action tools in the same turn as the check—wait for the exists result.",
                "Use move_mouse only when the task explicitly asks to move the cursor or interact with a named/specific on-screen target (e.g. 'click on the object', '點選物件', 'click the Submit button'). For 'click on the object' or '點選物件', split into move_mouse then click.",
                "Do not call move_mouse when the task only describes an action at the current cursor and does not name a target (e.g. triple-click, double-click, scroll, type text, press a key)—call that action tool directly.",
                "Click tool mapping: 點擊 / 點選 / single click → click; 連按兩下 / double-click → double_click. Never use double_click for a normal 點擊.",
                "For move_mouse: put the primary target in instruction (e.g. 「資料夾」圖示). When the task lists nearby landmarks (附近有… / near …), pass them as nearby_objects (e.g. [\"「Edge」圖示\", \"「Copilot」圖示\"]) instead of only embedding them in instruction.",
                "For drag: put the source in start_instruction and the drop target in destination_instruction. When the task lists start landmarks (起點附近有…), pass them as start_nearby_objects; when it lists destination landmarks (附近有… / 終點附近有… / near …), pass them as destination_nearby_objects (e.g. start_nearby_objects=[\"「Desktop」文字\"], destination_nearby_objects=[\"「新增文字文件txt」文字\"]) instead of only embedding them in the instructions.",
                "For scroll: positive clicks scroll down (往下滑), negative scroll up; use roughly 3–10 per screen of content.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "brain_decide_action_2": [
        {
            "image_usage": "optional",
            "prompt": (
                "Now you need to decide the next action to take. Either call tool(s) to continue, "
                "or finish with a JSON status object (no tool calls). Current task: {task}\n\n"
            ),
            "instructions": [
                'When finishing (no more tools), return strict JSON only: '
                '{{"status":"completed"|"failed","reason":"<short explanation>"}}. '
                "No markdown, no prose outside the JSON object.",
                'Use status "completed" only when CurrentTaskGoal is satisfied.',
                'Use status "failed" when the goal cannot be achieved (for example a required '
                "click/move target is not on screen after tool failures, or no viable method remains).",
                "If the previous task is not executed, try new method to achieve the task.",
                "If the tool failed to execute, do not assume the task is completed. Try new method "
                "to achieve the task; if no viable method remains, finish with status failed.",
                "If check_object_exists was the previous tool: read exists and the full CurrentTaskGoal "
                "to decide whether to continue. For 如果畫面上有… / if … is visible: continue with "
                "follow-up tools only when exists=true; when exists=false, finish with status completed "
                "(conditional step satisfied, no further tools). For 如果畫面上沒有… / if … is not visible: "
                "continue only when exists=false; when exists=true, finish with status completed. "
                "Do not treat exists=false as completed unless the goal is one of those visibility conditionals.",
                "When continuing after check_object_exists, issue the normal tools for the remainder of "
                "the step (e.g. move_mouse then click).",
                "If the task can be examined by screenshot, use the screenshot to examine if the task "
                "is completed if the screenshot is provided.",
                "If the task cannot be examined by screenshot, then assume the task is completed if "
                "the tool is executed successfully.",
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
                "Pick the candidate index from Candidates whose label matches the Anchor. "
                "Each candidate row is [index N] <label> center=(x,y)（<relative neighbor clauses>）. "
                "center=(x,y) is the primary anchor's screen center in pixels. "
                "Neighbor clauses describe the two nearest other detections (including nearby "
                "landmarks) with Traditional Chinese direction and pixel distance "
                "(左方/右方/上方/下方 + N個像素有<label>). "
                "Only rows in Candidates are selectable; Nearby labels and neighbor clauses "
                "are for disambiguation only.\n\n"
                "Anchor:\n{instruction}\n\n"
                "Nearby:\n{nearby_text}\n\n"
                "Candidates:\n{candidates_text}"
            ),
            "instructions": [
                'Reply only with JSON: {{"index": <integer>, "text": "<string>"}}.',
                '"index" is the [index] from the chosen candidate row (0-based).',
                '"text" must be that same row\'s text context copied verbatim after [index N] '
                "(label, optional center=(x,y), and neighbor clauses when present).",
                "Every Candidates row already matches the Anchor.",
                "When multiple candidates share the same Anchor label, prefer the one whose "
                "neighbor clauses best match the Nearby landmarks.",
                "Nearby may be (none); when it is, pick the best spatial match among Candidates.",
                "Never pick a Nearby landmark itself — only Candidates indices are valid.",
                "Never invent an index; only use an index shown in the Candidates list.",
                "index and text must describe the same Candidates row.",
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
                "Split candidates into two lists: those matching the Anchor, and those matching any Nearby label.\n"
                "Each row has class=文字(Text)|元素(Element)|未知(Unknown)|輸入欄(Input)|滾動條(Scrollbar). "
                "文字/未知 may include OCR text; 元素 uses icon labels only (no OCR text).\n\n"
                "Anchor:\n{anchor}\n\n"
                "Nearby:\n{nearby_text}\n\n"
                "Candidates:\n{candidates_text}\n"
            ),
            "instructions": [
                'Return JSON only: {{"anchor_indices": [<int>, ...], "nearby_indices": [<int>, ...]}}.',
                "Use [index N] values from the Candidates list. Use empty lists when none match.",
                "anchor_indices: every candidate whose OCR text, icons, or class matches the Anchor.",
                "nearby_indices: every candidate that matches any Nearby label "
                "(文字/圖示/元素/輸入欄/滾動條), for spatial disambiguation later.",
                "Do not put the same index in both lists; if a candidate matches both, put it only in "
                "anchor_indices.",
                "Nearby may be empty; when it is, nearby_indices must be [].",
                "Prefer recall: include all plausible matches for each list, not only the single best one.",
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
    "recording_action_to_cache": [
        {
            "image_usage": "optional",
            "prompt": (
                "You convert one recorded desktop user action into a single hub script "
                "instruction line.\n\n"
                "RecordedEvent:\n{event_json}\n\n"
                "Window state change detected (if any):\n{window_change_hint}\n\n"
                "Vision hints (screenshot pixels):\n"
                "Click location: [{cursor_x},{cursor_y}]\n"
                "Input field context:\n{field_context}\n"
                "Eight UI candidates nearest the click (closest first):\n"
                "Each row has class=文字(Text)|元素(Element)|輸入欄(Input)|滾動條(Scrollbar). "
                "文字 may include text='...' OCR; 元素 uses icons=Chinese icon labels only "
                "(no OCR text).\n\n"
                "{candidate_text}\n\n"
                "Drag destination vision (when kind=drag):\n"
                "Destination location: [{destination_x},{destination_y}]\n"
                "Destination field context:\n{destination_field_context}\n"
                "Eight UI candidates nearest the destination (closest first):\n\n"
                "{destination_candidate_text}\n\n"
                "Click/destination offset from cursor relative to each candidate center "
                "(dx=right, dy=down):\n"
                "{destination_offset_hints}\n"
            ),
            "instructions": [
                "Write one concise instruction in Traditional Chinese when possible, matching hub script style.",
                "Use the nearest matching candidate row to name the target (OCR text and/or icon labels).",
                "When Input field context shows visible text inside an 輸入欄(Input), name the target with that text plus 輸入欄, e.g. 將滑鼠移到「間間Gemini」文字所在的輸入欄. When the nearest candidate is an empty 輸入欄(Input) with no visible OCR text, name it as 輸入欄 (do not fall back to a nearby icon or text).",
                "For pointer clicks (click, double_click, right_click, middle_click): return only the move-target portion starting with 將滑鼠移到. Do not include nearby comments or the trailing action phrase; post-processing adds those. If the click is on the anchor, e.g. 將滑鼠移到「Submit」按鈕, 將滑鼠移到「間間Gemini」文字所在的輸入欄, or 將滑鼠移到輸入欄. If the click is off the anchor, append the relative pixel offset from the offset hints, e.g. 將滑鼠移到「自訂Office 範本」文字下方39個像素的位置. Use the exact pixel counts from the hints. Do not use vague 附近 for the primary target when an offset is available.",
                "For drag events: name the start target and destination anchor from candidates. Append the relative pixel offset from the offset hints to the destination anchor, e.g. 從「Chrome」圖示拖到「Desktop」文字下方49個像素的位置. Use the exact pixel counts from the hints for the anchor you name. If offset is negligible, omit it. If destination candidates are empty, describe the drop area directionally (e.g. 右側空白區域).",
                "For scroll: describe direction and target area, e.g. 在檔案清單區域向下捲動.",
                "For special keys: e.g. 按下 Enter 鍵.",
                "For modifier combos: e.g. 按下 Ctrl+C.",
                "For text_input events, another pipeline step writes 輸入「...」 directly; do not handle text_input here.",
                "If window state change hints report a confident minimize/maximize/close/restore action, prefer phrasing like 最小化/最大化/關閉/還原「title」視窗.",
                "Do not include absolute screen pixel coordinates in the instruction; relative pixel offsets for click targets and drag destinations are allowed.",
                'Return strict JSON only: {{"instruction": "<string>"}}',
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "recording_action_to_cache_retry": [
        {
            "image_usage": "optional",
            "prompt": (
                'Reply with ONLY: {{"instruction": "<string>"}}. '
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "recording_text_meaningful_check": [
        {
            "image_usage": "required",
            "prompt": (
                "You judge whether recorded keyboard text matches what the user intended to type.\n\n"
                "Recorded text from low-level key capture:\n{recorded_text}\n\n"
                "The attached screenshot shows the UI after typing finished.\n"
                "Return JSON only: {{\"meaningful\": <bool>, \"reason\": \"<short explanation>\"}}\n"
            ),
            "instructions": [
                "meaningful=true when recorded text is real user content: English words, Chinese characters, numbers, emails, URLs, or other intentional input.",
                "meaningful=false for IME composition keys only (pinyin like nihao without matching Chinese in the field), vk_* virtual-key tokens, or gibberish that clearly does not match visible field content.",
                "When the screenshot shows Chinese (or other composed text) in the focused field but recorded text is only Latin IME keys, meaningful=false.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "recording_text_meaningful_check_retry": [
        {
            "image_usage": "required",
            "prompt": (
                'Reply with ONLY: {{"meaningful": <bool>, "reason": "<short explanation>"}}. '
                "No text before or after the JSON."
            ),
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
                'Reply with ONLY: {{"anchor_indices": [<integer>, ...], '
                '"nearby_indices": [<integer>, ...]}}. '
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "instruction_relative_offset": [
        {
            "image_usage": "no_image",
            "prompt": (
                "Extract the UI anchor, relative pixel offset, and nearby labels from a "
                "mouse-target instruction.\n\n"
                "Instruction:\n{instruction}\n"
            ),
            "instructions": [
                'Return JSON only: {{"anchor": "<string>", "dx": <integer>, "dy": <integer>, "nearby": ["<string>", ...]}}.',
                "anchor: the on-screen target phrase for locating a UI element, without relative pixel offset clauses and without trailing 的位置.",
                "Keep quoted labels and type suffixes when present, e.g. 「振銓」文字, 「Chrome」圖示, 「Submit」按鈕.",
                "dx: horizontal offset in pixels from the anchor center; positive means right (右方), negative means left (左方).",
                "dy: vertical offset in pixels from the anchor center; positive means down (下方), negative means up (上方).",
                "Convert phrases like 右方5個像素 to dx=5, 上方28個像素 to dy=-28, 下方57個像素 to dy=57.",
                "When no relative pixel offset is stated, use dx=0 and dy=0.",
                "If the instruction is a drag sentence (從…拖到…), extract only the destination target and its offset.",
                "Ignore trailing （...） contextual comments when extracting anchor and offsets.",
                "Ignore inline （起點附近...） comments between the drag source and 拖到 when extracting anchor and offsets.",
                "nearby: list of nearby UI labels from contextual comments such as （附近有…）, （起點附近有…）, （終點附近有…）. "
                "Keep each label as written, including quoted names and type suffixes "
                "(文字/圖示/元素/輸入欄/滾動條/按鈕), e.g. 「圖片」文字, 「Chrome」圖示, 輸入欄, 滾動條.",
                "Preserve order of appearance. Use an empty list when no nearby labels are stated.",
                "Do not invent targets, offsets, or nearby labels not stated in the instruction.",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "instruction_relative_offset_retry": [
        {
            "image_usage": "no_image",
            "prompt": (
                'Reply with ONLY: {{"anchor": "<string>", "dx": <integer>, "dy": <integer>, '
                '"nearby": ["<string>", ...]}}. '
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_element_selection_retry": [
        {
            "image_usage": "no_image",
            "prompt": (
                'Reply with ONLY: {{"index": <integer>, "text": "<string>"}} - "index" is the '
                "[index] from the Candidates list row that best matches the location instruction "
                '(0-based), and "text" is that same row copied verbatim after [index N] '
                "(label, optional center=(x,y), and neighbor clauses when present). "
                "No text before or after the JSON."
            ),
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
}
