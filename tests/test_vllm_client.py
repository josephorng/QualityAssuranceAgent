from src.common.vllm_client import (
    _parse_call_syntax_arguments,
    _parse_call_syntax_tool_calls,
    _translate_openai_message_to_ollama,
)


def test_parse_call_syntax_arguments() -> None:
    args = _parse_call_syntax_arguments(
        "instruction:Minimize all open windows.,window_title_contains:all"
    )
    assert args == {
        "instruction": "Minimize all open windows.",
        "window_title_contains": "all",
    }


def test_parse_call_syntax_arguments_parses_nearby_objects_list() -> None:
    args = _parse_call_syntax_arguments(
        "instruction:「資料夾」圖示,nearby_objects:[「Edge」圖示,「Copilot」圖示]"
    )
    assert args == {
        "instruction": "「資料夾」圖示",
        "nearby_objects": ["「Edge」圖示", "「Copilot」圖示"],
    }


def test_parse_call_syntax_arguments_parses_json_list() -> None:
    args = _parse_call_syntax_arguments(
        'instruction:folder icon,nearby_objects:["Edge icon","Copilot icon"]'
    )
    assert args == {
        "instruction": "folder icon",
        "nearby_objects": ["Edge icon", "Copilot icon"],
    }


def test_parse_call_syntax_tool_calls() -> None:
    calls = _parse_call_syntax_tool_calls(
        "call:minimize_windows{instruction:Minimize all,window_title_contains:all}"
    )
    assert len(calls) == 1
    assert calls[0].function.name == "minimize_windows"
    assert calls[0].function.arguments == {
        "instruction": "Minimize all",
        "window_title_contains": "all",
    }


def test_parse_call_syntax_tool_calls_nearby_objects() -> None:
    calls = _parse_call_syntax_tool_calls(
        "call:move_mouse{instruction:「資料夾」圖示,nearby_objects:[「Edge」圖示,「Copilot」圖示]}"
    )
    assert len(calls) == 1
    assert calls[0].function.name == "move_mouse"
    assert calls[0].function.arguments == {
        "instruction": "「資料夾」圖示",
        "nearby_objects": ["「Edge」圖示", "「Copilot」圖示"],
    }


def test_translate_openai_message_parses_call_syntax() -> None:
    msg = _translate_openai_message_to_ollama(
        {
            "role": "assistant",
            "content": "call:wait{seconds:2,instruction:pause}",
            "tool_calls": [],
        }
    )
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].function.name == "wait"
    assert msg.tool_calls[0].function.arguments["seconds"] == "2"
