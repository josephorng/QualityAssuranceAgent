"""LLM 提示词定义。每个键映射到变体列表（使用第一个变体）。"""

from __future__ import annotations

from typing import Any

PROMPTS: dict[str, list[dict[str, Any]]] = {
    "brain_decide_action": [
        {
            "prompt": (
                "根据任务目标、当前截图和可用工具，"
                "决定一个或多个工具调用来完成任务目标。\n\n"
                "当前任务目标：\n{task}"
            ),
            "instructions": [
                "工具调用应按正确顺序执行，以完成任务目标。",
                "为每个工具调用编写详细的工具指令。",
                "所有显示器截图均已捕获，并将提供给你。",
                "不要执行任务范围以外的任何操作。",
                "滚动：正数向下滚动（往下滑），负数向上滚动；每屏内容大约使用 3–10 次。",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "brain_decide_action_2": [
        {
            "prompt": (
                "现在你需要决定下一步要执行的操作。如果任务已完成，"
                "请说明完成的原因。当前任务：{task}\n\n"
                "所有显示器截图均已捕获，并将提供给你。"
            ),
            "instructions": [
                "如果上一步任务未执行，请尝试新方法来完成任务。",
                "如果工具执行失败，不要假定任务已完成。请尝试新方法来完成任务。",
                "如果任务可以通过截图验证，请使用截图检查任务是否已完成。",
                "如果任务无法通过截图验证，则在工具成功执行后假定任务已完成。",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "brain_verify_script_step": [
        {
            "prompt": (
                "你正在验证当前脚本任务步骤是否在截图中已满足。"
                "你将看到完整的编号脚本以及当前步骤。"
            ),
            "instructions": [
                "根据可见的 UI 和文字，判断当前步骤目标是否已在屏幕上实际完成。",
                '仅返回严格 JSON（无 markdown），单个对象，键为：accomplished (bool)、branch (string)、target_step (number 或 null)、reason (string)。',
                "branch 必须是以下之一：advance、retry、skip、goto。",
                "仅当 accomplished 为 true 时使用 branch advance（进入下一脚本行）。",
                "当 accomplished 为 false 时：使用 retry 重复当前步骤，skip 放弃本行并进入下一行，或 goto 跳转到指定脚本行（target_step 为编号列表中的 1 基行号）。",
                "对于 goto，target_step 必须是每行脚本前显示的行号（1 到 N）。其他分支将 target_step 设为 null。",
                "不要臆造 UI 元素；仅根据图像和脚本文本得出结论。",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "coordinate_selection": [
        {
            "prompt": (
                "从 CoordinatesText 中选择一行，使其最匹配 Target。\n"
                "CoordinatesText 每行格式为：[center_x,center_y] <该区域的 OCR 文本>。\n\n"
                "Target：\n{target}\n\n"
                "Instruction：\n{instruction}\n\n"
                "CoordinatesText：\n{coordinate_text}\n"
            ),
            "instructions": [
                "OCR 文本可能有错别字和错误，请仔细匹配文本。",
                "仅回复 OCR 文本（方括号后的部分），尽可能从 CoordinatesText 逐字复制，以便即使存在错别字也能匹配。",
                "除符合服务端 schema 的有效 JSON 外，不要输出任何内容。",
                "不要总结、分类、列表、markdown、翻译、解释、添加键或添加说明文字。",
                "仅返回严格 JSON。",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "coordinate_disambiguation": [
        {
            "prompt": (
                "匹配的 OCR 文本在图像中出现于多个位置。\n"
                "选择一个中心点 (x, y)，使其最匹配 Instruction。\n"
                "(x, y) 必须是下方列出的候选中心之一——"
                "与 CoordinatesText 相同的坐标空间（图像像素）。\n"
                '"text" 必须是同一选择的 OCR 行：从你选择的中心对应的 [cx,cy] 后复制（尽可能逐字；OCR 可能有错别字）。\n\n'
                "Instruction：\n{instruction}\n\n"
                "第一步匹配的文本：\n{chosen_text}\n\n"
                "候选中心（请恰好选择一个）：\n{options_lines}\n"
            ),
            "instructions": [
                "请注意 OCR 文本可能有错别字和错误，请仔细匹配文本。",
                "除符合服务端 schema 的有效 JSON 外，不要输出任何内容。",
                "不要总结、解释或添加说明文字。",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_element_selection": [
        {
            "prompt": (
                "从 Candidates 中选择最匹配 Instruction 位置提示的候选索引。"
                "每行候选以 [index] 开头，然后是 center=[cx,cy] w=<width_px> h=<height_px>。\n\n"
                "Instruction：\n{instruction}\n\n"
                "Candidates：\n{candidates_text}\n"
            ),
            "instructions": [
                '仅回复 JSON：{{"index": <integer>}}——所选候选行的 [index]（从 0 开始）。',
                "不要编造索引；仅使用 Candidates 列表中显示的索引。",
                "不要添加额外文字或解释。",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
    "ui_instruction_icon_location_extract": [
        {
            "prompt": (
                "分析以下 UI 自动化指令，在一次回复中为下游模型提取信息。\n\n"
                "用户指令：\n{instruction}\n"
            ),
            "instructions": [
                '仅返回 JSON：{{"need_text_anchor": <true|false>, "location_description": "<string>"}}。',
                "need_text_anchor：当指令涉及可见文字、标签或屏幕文本内容时设为 true（例如：点击「登录」、名为 X 的行、按标题选择）。当目标主要是非文本视觉元素（图标、开关、头像、齿轮、无标签按钮、面板）且无实质文本锚点时设为 false。",
                "location_description：用于区分多个屏幕候选的详细空间描述：区域（上/下/左/右/中、角落）、相对布局（上方/下方/旁边/邻近）、序数（第一/最后一行）、与窗口边缘的距离、标题栏/页脚/工具栏/侧边栏（如有暗示）。将模糊提示展开为明确的位置语言。若无位置线索，使用空字符串。",
                "不要臆造指令中未暗示的 UI。",
                "不要在 JSON 对象之外输出 markdown 或说明文字。",
            ],
            "models": ["gemma4:e2b", "gemma3:4b"],
        }
    ],
}
