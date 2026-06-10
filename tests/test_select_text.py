from __future__ import annotations

from cua_mcp.icon_map import map_pua_in_text
from cua_mcp.select_text import (
    _best_row_by_llm_text_similarity,
    _matches_without_pua,
    _regions_with_mapped_pua,
    _strip_pua_from_text,
)


def test_strip_pua_from_text() -> None:
    assert _strip_pua_from_text("Sign\uE000 in") == "Sign in"
    assert _strip_pua_from_text("\uf000") == ""


def test_map_pua_in_text_uses_icon_map_chinese_id() -> None:
    folder_pua = "\ue000"
    assert map_pua_in_text(folder_pua) == "資料夾"
    assert map_pua_in_text(f"Open{folder_pua}") == "Open資料夾"
    assert map_pua_in_text("plain text") == "plain text"
    assert map_pua_in_text("\uf000") == "未知圖示"


def test_regions_with_mapped_pua() -> None:
    regions = [
        (((0, 0, 10, 10), (5, 5), ["\ue000"])),
        (((0, 0, 10, 10), (15, 15), ["OK"])),
    ]
    mapped = _regions_with_mapped_pua(regions)
    assert mapped[0][2] == ["資料夾"]
    assert mapped[1][2] == ["OK"]


def test_best_row_by_llm_text_similarity_prefers_containing_row() -> None:
    matches = [
        (1558, 1020, "麵無"),
        (1978, 245, "空心方框 桌面"),
        (2446, 200, "麵。有帳"),
    ]
    assert _best_row_by_llm_text_similarity("桌面", matches) == (1978, 245, "空心方框 桌面")


def test_matches_without_pua_drops_empty_and_keeps_labels() -> None:
    matches = [
        (10, 20, "OK"),
        (30, 40, "\ue001"),
        (50, 60, "OK\uE002"),
    ]
    assert _matches_without_pua(matches) == [(10, 20, "OK"), (50, 60, "OK")]
