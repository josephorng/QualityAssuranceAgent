from __future__ import annotations

import json

from PIL import Image

from app_ocr_verify_tk import (
    OcrLine,
    TextVerifyResult,
    _format_inference_timing,
    build_verify_sheet_image,
    compare_readings_to_ocr,
    export_element_line_to_dir,
    export_label_for_verified_line,
    export_text_line_to_dir,
    gemma_label_for_verified_line,
    is_unknown_element_line,
    parse_sheet_read_response,
    text_lines_from_yolo,
    unknown_element_lines_from_yolo,
    visible_line_indices_for_filter,
)
from cua_mcp.yolo_onnx import YOLO_CLASS_ELEMENT, YOLO_CLASS_TEXT
from app_ocr_viewer_tk import _unknown_icon_label


def test_text_lines_from_yolo_filters_non_text_classes() -> None:
    lines = [
        OcrLine(box=(0, 0, 10, 10), text="hello", class_name="text", class_id=0),
        OcrLine(box=(1, 1, 10, 10), text="icon", class_name="element", class_id=1),
        OcrLine(box=(2, 2, 10, 10), text="also text", class_name="text"),
    ]
    filtered = text_lines_from_yolo(lines)
    assert [line.text for line in filtered] == ["hello", "also text"]


def test_build_verify_sheet_image_numbers_rows() -> None:
    source = Image.new("RGB", (100, 100), color=(255, 255, 255))
    lines = [
        OcrLine(box=(5, 5, 20, 10), text="a"),
        OcrLine(box=(5, 30, 20, 10), text="b"),
    ]
    sheet = build_verify_sheet_image(source, lines, start_index=3)
    assert sheet.width > 0
    assert sheet.height > 0


def test_crop_line_image_expands_bbox() -> None:
    from app_ocr_verify_tk import OCR_VERIFY_BOX_EXPAND, _crop_line_image

    source = Image.new("RGB", (50, 50), color=(0, 0, 0))
    tight = _crop_line_image(source, (10, 10, 10, 10), expand=0)
    expanded = _crop_line_image(source, (10, 10, 10, 10), expand=OCR_VERIFY_BOX_EXPAND)
    assert expanded.size[0] == tight.size[0] + 2 * OCR_VERIFY_BOX_EXPAND
    assert expanded.size[1] == tight.size[1] + 2 * OCR_VERIFY_BOX_EXPAND


def test_compare_readings_to_ocr_marks_mismatches() -> None:
    lines = [
        OcrLine(box=(0, 0, 1, 1), text="hello"),
        OcrLine(box=(1, 1, 1, 1), text="world"),
    ]
    results = compare_readings_to_ocr(lines, {0: "hello", 1: "World"}, start_index=0)
    assert results[0].correct is True
    assert results[1].correct is False
    assert results[1].expected_text == "World"


def test_parse_sheet_read_response() -> None:
    payload = {
        "readings": [{"index": 2, "text": "foo"}],
        "summary": "ok",
    }
    readings, summary = parse_sheet_read_response(json.dumps(payload))
    assert summary == "ok"
    assert readings == {2: "foo"}


def test_export_label_uses_gemma_reading() -> None:
    line = OcrLine(box=(0, 0, 1, 1), text="wrng")
    incorrect = TextVerifyResult(
        index=0,
        recognized_text="wrng",
        correct=False,
        expected_text="right",
    )
    assert export_label_for_verified_line(line, incorrect) == "right"
    assert gemma_label_for_verified_line(line, incorrect) == "right"

    correct_line = OcrLine(box=(0, 0, 1, 1), text="hello")
    correct = TextVerifyResult(
        index=0,
        recognized_text="hello",
        correct=True,
        expected_text="",
    )
    assert export_label_for_verified_line(correct_line, correct) == "hello"

    assert export_label_for_verified_line(line, None) == ""
    assert export_label_for_verified_line(line, incorrect, override="edited") == "edited"


def test_copy_image_to_undone_uses_viewer_naming(tmp_path, monkeypatch) -> None:
    from app_ocr_verify_tk import copy_image_to_undone

    monkeypatch.setattr(
        "app_ocr_verify_tk.YOLO_UNDONE_IMAGES",
        tmp_path,
    )
    src = tmp_path / "src" / "shot.png"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"png")
    dest = copy_image_to_undone(src, "my_run")
    assert dest == tmp_path / "cua_my_run_shot.png"
    assert dest.read_bytes() == b"png"


def test_unknown_element_lines_from_yolo() -> None:
    unknown = _unknown_icon_label()
    lines = [
        OcrLine(box=(0, 0, 1, 1), text="hello", class_name="text", class_id=YOLO_CLASS_TEXT),
        OcrLine(
            box=(1, 1, 1, 1),
            text="",
            class_name="element",
            class_id=YOLO_CLASS_ELEMENT,
            chinese_ids=(unknown,),
        ),
        OcrLine(
            box=(2, 2, 1, 1),
            text="",
            class_name="element",
            class_id=YOLO_CLASS_ELEMENT,
            chinese_ids=("save",),
        ),
    ]
    assert is_unknown_element_line(lines[1]) is True
    assert is_unknown_element_line(lines[2]) is False
    filtered = unknown_element_lines_from_yolo(lines)
    assert [line.chinese_ids for line in filtered] == [(unknown,)]


def test_export_element_line_to_dir_writes_png_only(tmp_path) -> None:
    image = Image.new("RGB", (20, 20), color=(255, 255, 255))
    line = OcrLine(box=(2, 3, 8, 6), text="", class_name="element", class_id=YOLO_CLASS_ELEMENT)
    paths = export_element_line_to_dir(
        image,
        line,
        tmp_path,
        base_name="shot",
        item_index=0,
    )
    assert len(paths) == 1
    assert paths[0].name == "shot_obj_item001.png"
    assert not (tmp_path / "shot_obj_item001.txt").exists()


def test_visible_line_indices_for_filter_defaults_to_mismatches() -> None:
    results = [
        TextVerifyResult(index=0, recognized_text="a", correct=True),
        TextVerifyResult(index=1, recognized_text="b", correct=False, expected_text="B"),
        TextVerifyResult(index=2, recognized_text="c", correct=False, expected_text="C"),
    ]
    assert visible_line_indices_for_filter(3, results, mismatch_only=True) == [1, 2]
    assert visible_line_indices_for_filter(3, results, mismatch_only=False) == [0, 1, 2]
    assert visible_line_indices_for_filter(3, None, mismatch_only=True) == [0, 1, 2]


def test_format_inference_timing_shows_both_phases() -> None:
    assert _format_inference_timing(ocr_elapsed_s=1.2, gemma_elapsed_s=8.4) == "OCR 1.2s | Gemma 8.4s"
    assert _format_inference_timing(ocr_elapsed_s=1.2) == "OCR 1.2s"
    assert _format_inference_timing(gemma_elapsed_s=8.4) == "Gemma 8.4s"
    assert _format_inference_timing() == ""


def test_export_text_line_to_dir_writes_png_and_txt(tmp_path) -> None:
    image = Image.new("RGB", (20, 20), color=(255, 255, 255))
    line = OcrLine(box=(2, 3, 8, 6), text="hello")
    paths = export_text_line_to_dir(
        image,
        line,
        "hello",
        tmp_path,
        base_name="shot",
        item_index=0,
    )
    assert len(paths) == 2
    assert paths[0].name == "shot_item001.png"
    assert paths[1].name == "shot_item001.txt"
    assert paths[1].read_text(encoding="utf-8") == "hello"
