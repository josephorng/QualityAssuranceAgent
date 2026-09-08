from src.brain.module import BrainModule
from src.common.llm_text import LLM_QUOTE_WRAPPER, strip_llm_quote_wrappers
from src.common.nearby_side import enrich_tool_arguments_from_goal


def test_strip_llm_quote_wrappers_string() -> None:
    assert strip_llm_quote_wrappers(f'{LLM_QUOTE_WRAPPER}輸入欄{LLM_QUOTE_WRAPPER}') == "輸入欄"
    assert strip_llm_quote_wrappers("plain") == "plain"
    assert strip_llm_quote_wrappers(f"left{LLM_QUOTE_WRAPPER}") == "left"


def test_strip_llm_quote_wrappers_nested_tool_args() -> None:
    cleaned = strip_llm_quote_wrappers(
        {
            "instruction": f'{LLM_QUOTE_WRAPPER}輸入欄{LLM_QUOTE_WRAPPER}',
            "nearby_objects": [
                f'{LLM_QUOTE_WRAPPER}在「圖片文件」圖示的左上方{LLM_QUOTE_WRAPPER}',
                f'{LLM_QUOTE_WRAPPER}在「帳號」文字的右邊{LLM_QUOTE_WRAPPER}',
                f'{LLM_QUOTE_WRAPPER}在「確定」文字的上面{LLM_QUOTE_WRAPPER}',
            ],
            "clicks": f"{LLM_QUOTE_WRAPPER}1{LLM_QUOTE_WRAPPER}",
            "nested": {"button": f'{LLM_QUOTE_WRAPPER}left{LLM_QUOTE_WRAPPER}'},
        }
    )
    assert cleaned == {
        "instruction": "輸入欄",
        "nearby_objects": [
            "在「圖片文件」圖示的左上方",
            "在「帳號」文字的右邊",
            "在「確定」文字的上面",
        ],
        "clicks": "1",
        "nested": {"button": "left"},
    }


def test_strip_llm_quote_wrappers_preserves_non_strings() -> None:
    assert strip_llm_quote_wrappers(None) is None
    assert strip_llm_quote_wrappers(3) == 3
    assert strip_llm_quote_wrappers(True) is True


def test_brain_normalize_tool_arguments_strips_wrappers() -> None:
    cleaned = BrainModule._normalize_tool_arguments(
        {
            "instruction": f'{LLM_QUOTE_WRAPPER}輸入欄{LLM_QUOTE_WRAPPER}',
            "nearby_objects": [
                f'{LLM_QUOTE_WRAPPER}在「帳號」文字的右邊{LLM_QUOTE_WRAPPER}',
            ],
        }
    )
    assert cleaned == {
        "instruction": "輸入欄",
        "nearby_objects": ["在「帳號」文字的右邊"],
    }


def test_strip_then_enrich_nearby_does_not_keep_wrapper_duplicates() -> None:
    """Regression: wrapped nearby copies must not survive into move_mouse args.

    Without stripping, goal-side restore kept both clean directed phrases and
    ``<|"|>…<|"|>`` undirected copies, so geometric prefilter could not uniquely
    pick the account input over the password field.
    """
    goal = (
        "將滑鼠移到輸入欄（在「圖片文件」圖示的左上方、在「帳號」文字的右邊、"
        "在「確定」文字的上面），並點擊滑鼠一下。"
    )
    args = BrainModule._normalize_tool_arguments(
        {
            "instruction": f'{LLM_QUOTE_WRAPPER}輸入欄{LLM_QUOTE_WRAPPER}',
            "nearby_objects": [
                f'{LLM_QUOTE_WRAPPER}在「圖片文件」圖示的左上方{LLM_QUOTE_WRAPPER}',
                f'{LLM_QUOTE_WRAPPER}在「帳號」文字的右邊{LLM_QUOTE_WRAPPER}',
                f'{LLM_QUOTE_WRAPPER}在「確定」文字的上面{LLM_QUOTE_WRAPPER}',
            ],
        }
    )
    enriched = enrich_tool_arguments_from_goal("move_mouse", args, goal)
    assert enriched["instruction"] == "輸入欄"
    assert enriched["nearby_objects"] == [
        "在「圖片文件」圖示的左上方",
        "在「帳號」文字的右邊",
        "在「確定」文字的上面",
    ]
    assert all(LLM_QUOTE_WRAPPER not in item for item in enriched["nearby_objects"])
