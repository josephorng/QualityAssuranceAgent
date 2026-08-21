from __future__ import annotations

import ast
import csv
import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

from src.common.script_helper import script_display_name
from src.common.session_report import _resolve_script_metadata, _resolve_started_at_utc

_HTML_NAME = "session_steps.html"
_RECORDING_HTML_NAME = "recording_steps.html"
_INDEX_HTML_NAME = "index.html"
_RUN_FOLDER_TS_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}_(\d{8})_(\d{6})_\d+$")
_QUEUE_SCRIPT_LOG_MARKER = "Queue starting coordinator for "
_HAND_OP_PREFIX = "動作 "
_UNGROUPED_GOAL = "未分類動作"
_FALLBACK_GOAL = "手部動作"
_RECORDING_KIND_LABELS = {
    "click": "點擊",
    "double_click": "雙擊",
    "triple_click": "連按3下",
    "right_click": "右鍵點擊",
    "middle_click": "中鍵點擊",
    "hold": "按住",
    "drag": "拖曳",
    "scroll": "捲動",
    "text_input": "輸入文字",
    "key": "按鍵",
    "key_press": "按鍵",
    "hotkey": "快捷鍵",
    "wait": "等待",
    "manual": "自訂指令",
}

# Column order written by ``src.hand.module`` via ``append_csv_row``; used as a fallback for
# older ``hand.csv`` files that were saved without a header row.
_HAND_CSV_FIELDS = [
    "timestamp",
    "action",
    "args",
    "ok",
    "screenshot_name",
    "screenshot_before_path",
    "screenshot_after_path",
    "message",
]

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Microsoft JhengHei", system-ui, sans-serif;
  margin: 0; padding: 2rem; line-height: 1.5;
  background: #f5f6f8; color: #1f2328;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
.intro { color: #57606a; margin: 0 0 1.5rem; }
.nav { margin: 0 0 1rem; }
.nav a { color: #0969da; text-decoration: none; font-weight: 600; }
.nav a:hover { text-decoration: underline; }
.instruction-group {
  background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
  padding: 0; margin: 0 0 1.25rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.instruction-group > summary {
  display: flex; align-items: center; justify-content: space-between; gap: .75rem;
  cursor: pointer; user-select: none; list-style: none;
  padding: 1rem 1.5rem; font-size: 1.05rem; font-weight: 600;
}
.instruction-group > summary::-webkit-details-marker { display: none; }
.instruction-group > summary::before {
  content: "▶"; flex: 0 0 auto; font-size: .7rem; color: #57606a;
  transition: transform .15s ease;
}
.instruction-group[open] > summary::before { transform: rotate(90deg); }
.instruction-group > summary .instruction-number {
  flex: 0 0 auto; color: #57606a; font-variant-numeric: tabular-nums;
}
.instruction-group > summary .instruction-title { flex: 1 1 auto; min-width: 0; }
.instruction-group > summary .instruction-summary-text {
  flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: .2rem;
}
.instruction-group > summary .instruction-summary-text .instruction-title {
  flex: 0 1 auto;
}
.instruction-group > summary .instruction-expected {
  font-size: .82rem; font-weight: 500; color: #57606a; line-height: 1.35;
}
.instruction-group > summary .instruction-expected-empty {
  color: #8c959f; font-style: italic;
}
.instruction-group > summary .instruction-badges {
  display: inline-flex; flex-wrap: wrap; gap: .35rem; flex: 0 0 auto;
}
.instruction-group[open] > summary { border-bottom: 1px solid #d0d7de; }
.copy-instruction {
  appearance: none; border: 1px solid #d0d7de; background: #f6f8fa;
  cursor: pointer; border-radius: 6px; padding: .2rem .55rem;
  font-size: .75rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #57606a; flex: 0 0 auto;
}
.copy-instruction:hover { background: #eaeef2; color: #1f2328; }
.copy-instruction.copied {
  color: #116329; border-color: #4ac26b; background: #dafbe1;
}
.delete-instruction {
  appearance: none; border: 1px solid #d0d7de; background: #f6f8fa;
  cursor: pointer; border-radius: 6px; padding: .2rem .55rem;
  font-size: .75rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #57606a; flex: 0 0 auto;
}
.delete-instruction:hover { background: #ffebe9; color: #cf222e; border-color: #ff8182; }
.delete-instruction:disabled { opacity: .45; cursor: not-allowed; }
.add-instruction {
  appearance: none; border: 1px solid #d0d7de; background: #f6f8fa;
  cursor: pointer; border-radius: 6px; padding: .2rem .55rem;
  font-size: .75rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #57606a; flex: 0 0 auto;
}
.add-instruction:hover { background: #ddf4ff; color: #0969da; border-color: #54aeff; }
.collapse-row {
  display: flex; justify-content: center; align-items: center;
  padding: 0 1.5rem .85rem;
}
.collapse-instruction {
  appearance: none; border: 1px solid #d0d7de; background: #f6f8fa;
  cursor: pointer; border-radius: 6px; padding: .25rem .7rem;
  font-size: .7rem; line-height: 1; font-family: inherit;
  color: #57606a; flex: 0 0 auto;
}
.collapse-instruction:hover { background: #eaeef2; color: #1f2328; }
.recording-toolbar {
  display: flex; flex-wrap: wrap; align-items: center; gap: .5rem;
  margin: 0 0 1.25rem;
}
.copy-all-instructions {
  appearance: none; border: 1px solid #d0d7de; background: #f6f8fa;
  cursor: pointer; border-radius: 6px; padding: .35rem .75rem;
  font-size: .85rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #57606a;
}
.copy-all-instructions:hover:not(:disabled) { background: #eaeef2; color: #1f2328; }
.copy-all-instructions:disabled { opacity: .45; cursor: not-allowed; }
.copy-all-instructions.copied {
  color: #116329; border-color: #4ac26b; background: #dafbe1;
}
.rename-recording {
  appearance: none; border: 1px solid #d0d7de; background: #f6f8fa;
  cursor: pointer; border-radius: 6px; padding: .35rem .75rem;
  font-size: .85rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #57606a;
}
.rename-recording:hover:not(:disabled) { background: #eaeef2; color: #1f2328; }
.add-recording-step {
  appearance: none; border: 1px solid #0969da; background: #ddf4ff;
  cursor: pointer; border-radius: 6px; padding: .35rem .75rem;
  font-size: .85rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #0969da;
}
.add-recording-step:hover { background: #b6e3ff; }
.landmarks {
  margin: 0 1.5rem 1rem; padding: .75rem 1rem;
  border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa;
}
.landmarks-title {
  margin: 0 0 .5rem; font-size: .9rem; font-weight: 700; color: #57606a;
}
.landmarks-groups {
  display: flex; flex-wrap: wrap; align-items: flex-start;
  gap: .85rem 1.25rem; margin: 0 0 .65rem;
}
.landmarks-group {
  margin: 0; flex: 1 1 14rem; min-width: 12rem; max-width: 100%;
}
.landmarks-group-label {
  margin: 0 0 .35rem; font-size: .8rem; font-weight: 600; color: #8c959f;
}
.landmarks-side-groups {
  display: flex; flex-direction: column; gap: .55rem;
}
.landmarks-side-group { margin: 0; }
.landmarks-side-label {
  margin: 0 0 .2rem; font-size: .75rem; font-weight: 700; color: #57606a;
}
.landmarks-list {
  list-style: none; margin: 0; padding: 0;
  display: flex; flex-direction: column; gap: .25rem;
}
.landmarks-list label {
  display: flex; align-items: flex-start; gap: .45rem;
  font-size: .85rem; font-weight: 500; cursor: pointer; color: #1f2328;
}
.landmarks-list input { margin-top: .2rem; flex: 0 0 auto; }
.apply-landmarks {
  appearance: none; border: 1px solid #0969da; background: #ddf4ff;
  cursor: pointer; border-radius: 6px; padding: .3rem .7rem;
  font-size: .8rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #0969da;
}
.apply-landmarks:hover:not(:disabled) { background: #b6e3ff; }
.apply-landmarks:disabled { opacity: .45; cursor: not-allowed; }
.apply-landmarks.applied {
  color: #116329; border-color: #4ac26b; background: #dafbe1;
}
.landmarks-status {
  display: inline-block; margin-left: .5rem;
  font-size: .75rem; color: #57606a; font-weight: 600;
}
.landmarks-status.error { color: #cf222e; }
.vision-retry {
  margin: 0 1.5rem 1rem; padding: .75rem .9rem;
  border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa;
}
.vision-retry.failed { border-color: #eac54f; background: #fff8c5; }
.vision-retry-title { font-weight: 600; margin: 0 0 .35rem; }
.vision-retry-note { margin: 0 0 .6rem; color: #57606a; font-size: .85rem; }
.rerun-yolo-ocr {
  appearance: none; border: 1px solid #0969da; background: #ddf4ff;
  cursor: pointer; border-radius: 6px; padding: .3rem .7rem;
  font-size: .8rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #0969da;
}
.rerun-yolo-ocr:hover:not(:disabled) { background: #b6e3ff; }
.rerun-yolo-ocr:disabled { opacity: .45; cursor: not-allowed; }
.vision-retry-status {
  margin-left: .6rem; font-size: .8rem; font-weight: 600; color: #57606a;
}
.vision-retry-status.error { color: #cf222e; }
.typed-text {
  margin: 0 1.5rem 1rem; padding: .75rem 1rem;
  border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa;
}
.typed-text-title {
  margin: 0 0 .5rem; font-size: .9rem; font-weight: 700; color: #57606a;
}
.typed-text-choices {
  display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 .5rem;
}
.typed-text-choice {
  appearance: none; border: 1px solid #d0d7de; background: #fff;
  cursor: pointer; border-radius: 6px; padding: .25rem .55rem;
  font-size: .8rem; line-height: 1.3; font-family: inherit; color: #1f2328;
}
.typed-text-choice:hover { border-color: #0969da; background: #f6f8fa; }
.typed-text-choice.selected {
  border-color: #0969da; background: #ddf4ff; font-weight: 600; color: #0969da;
}
.typed-text-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: .5rem;
}
.typed-text-input {
  flex: 1 1 16rem; min-width: 12rem;
  font-family: inherit; font-size: .9rem; line-height: 1.3;
  padding: .35rem .55rem; border: 1px solid #d0d7de; border-radius: 6px;
  background: #fff; color: #1f2328;
}
.apply-typed-text {
  appearance: none; border: 1px solid #0969da; background: #ddf4ff;
  cursor: pointer; border-radius: 6px; padding: .3rem .7rem;
  font-size: .8rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #0969da; flex: 0 0 auto;
}
.apply-typed-text:hover:not(:disabled) { background: #b6e3ff; }
.apply-typed-text:disabled { opacity: .45; cursor: not-allowed; }
.apply-typed-text.applied {
  color: #116329; border-color: #4ac26b; background: #dafbe1;
}
.typed-text-status {
  display: inline-block;
  font-size: .75rem; color: #57606a; font-weight: 600;
}
.typed-text-status.error { color: #cf222e; }
.instruction-edit-panels {
  display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem; padding: 0 1.5rem; margin: 0 0 1rem;
}
.expected-outcome {
  margin: 0; padding: .75rem 1rem;
  border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa;
}
.expected-outcome-title {
  margin: 0 0 .5rem; font-size: .9rem; font-weight: 700; color: #57606a;
}
.expected-outcome-row {
  display: flex; flex-wrap: wrap; align-items: flex-start; gap: .5rem;
}
.expected-outcome-input {
  flex: 1 1 16rem; min-width: 12rem; min-height: 2.6rem; resize: vertical;
  font-family: inherit; font-size: .9rem; line-height: 1.3;
  padding: .35rem .55rem; border: 1px solid #d0d7de; border-radius: 6px;
  background: #fff; color: #1f2328;
}
.apply-expected-outcome {
  appearance: none; border: 1px solid #0969da; background: #ddf4ff;
  cursor: pointer; border-radius: 6px; padding: .3rem .7rem;
  font-size: .8rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #0969da; flex: 0 0 auto;
}
.apply-expected-outcome:hover:not(:disabled) { background: #b6e3ff; }
.apply-expected-outcome:disabled { opacity: .45; cursor: not-allowed; }
.apply-expected-outcome.applied {
  color: #116329; border-color: #4ac26b; background: #dafbe1;
}
.expected-outcome-status {
  display: inline-block;
  font-size: .75rem; color: #57606a; font-weight: 600;
}
.expected-outcome-status.error { color: #cf222e; }
.step-instruction {
  margin: 0; padding: .75rem 1rem;
  border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa;
}
.step-instruction-title {
  margin: 0 0 .5rem; font-size: .9rem; font-weight: 700; color: #57606a;
}
.step-instruction-row {
  display: flex; flex-wrap: wrap; align-items: flex-start; gap: .5rem;
}
.step-instruction-input {
  flex: 1 1 16rem; min-width: 12rem; min-height: 2.6rem; resize: vertical;
  font-family: inherit; font-size: .9rem; line-height: 1.3;
  padding: .35rem .55rem; border: 1px solid #d0d7de; border-radius: 6px;
  background: #fff; color: #1f2328;
}
.apply-step-instruction {
  appearance: none; border: 1px solid #0969da; background: #ddf4ff;
  cursor: pointer; border-radius: 6px; padding: .3rem .7rem;
  font-size: .8rem; line-height: 1.2; font-family: inherit;
  font-weight: 600; color: #0969da; flex: 0 0 auto;
}
.apply-step-instruction:hover:not(:disabled) { background: #b6e3ff; }
.apply-step-instruction:disabled { opacity: .45; cursor: not-allowed; }
.apply-step-instruction.applied {
  color: #116329; border-color: #4ac26b; background: #dafbe1;
}
.step-instruction-status {
  display: inline-block;
  font-size: .75rem; color: #57606a; font-weight: 600;
}
.step-instruction-status.error { color: #cf222e; }
.add-step-dialog-backdrop {
  display: none; position: fixed; inset: 0; z-index: 40;
  align-items: center; justify-content: center;
  background: rgba(31,35,40,.45); padding: 1.25rem;
}
.add-step-dialog-backdrop.open { display: flex; }
.add-step-dialog {
  width: min(34rem, 100%); max-height: 90vh; overflow: auto;
  background: #fff; color: #1f2328; border-radius: 10px;
  border: 1px solid #d0d7de; box-shadow: 0 8px 24px rgba(0,0,0,.12);
  padding: 1.15rem 1.35rem 1.25rem;
}
.add-step-dialog h2 {
  margin: 0 0 .85rem; font-size: 1.1rem;
}
.add-step-field { margin: 0 0 .75rem; }
.add-step-field label {
  display: block; margin: 0 0 .3rem;
  font-size: .8rem; font-weight: 700; color: #57606a;
}
.add-step-field input,
.add-step-field select,
.add-step-field textarea {
  width: 100%; font-family: inherit; font-size: .9rem;
  padding: .35rem .55rem; border: 1px solid #d0d7de; border-radius: 6px;
  background: #fff; color: #1f2328;
}
.add-step-field textarea { min-height: 4.2rem; resize: vertical; }
.add-step-actions {
  display: flex; justify-content: flex-end; gap: .5rem; margin-top: 1rem;
}
.add-step-actions button {
  appearance: none; border-radius: 6px; padding: .35rem .75rem;
  font-size: .85rem; line-height: 1.2; font-family: inherit; font-weight: 600;
  cursor: pointer;
}
.add-step-cancel {
  border: 1px solid #d0d7de; background: #f6f8fa; color: #57606a;
}
.add-step-cancel:hover { background: #eaeef2; color: #1f2328; }
.add-step-submit {
  border: 1px solid #0969da; background: #ddf4ff; color: #0969da;
}
.add-step-submit:hover:not(:disabled) { background: #b6e3ff; }
.add-step-submit:disabled { opacity: .45; cursor: not-allowed; }
.add-step-status {
  margin: .25rem 0 0; font-size: .8rem; font-weight: 600; color: #57606a;
}
.add-step-status.error { color: #cf222e; }
.hand-ops {
  list-style: disc; margin: 0; padding: 1rem 1.5rem 1.25rem 2.5rem;
}
.hand-op { margin: 0 0 1.25rem; }
.hand-op:last-child { margin-bottom: 0; }
.hand-op-title {
  display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
  font-weight: 600; margin: 0 0 .75rem;
}
.hand-op-action {
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
}
.smart-cycle-meta { padding: 1rem 1.5rem 0; }
.session-verify {
  margin: 1rem 1.5rem 0; padding: .75rem 1rem;
  border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa;
}
.session-verify-title {
  margin: 0 0 .5rem; font-size: .9rem; font-weight: 700; color: #57606a;
}
.session-verify .meta { margin: 0; }
.executed-tools { border-top: 1px solid #d0d7de; }
.executed-tools h3 { font-size: 1rem; margin: 0; padding: 1rem 1.5rem 0; }
.meta { margin: 0 0 1rem; }
.meta dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem; margin: 0; }
.meta dt { color: #57606a; font-weight: 600; }
.meta dd { margin: 0; word-break: break-word; white-space: pre-wrap; }
.badge { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.badge.ok { background: #dafbe1; color: #116329; }
.badge.fail { background: #ffebe9; color: #cf222e; }
.badge.neutral { background: #eaeef2; color: #57606a; }
.args { margin: 0 0 1rem; border: 1px solid #d0d7de; border-radius: 6px; background: #f6f8fa; }
.args > summary { cursor: pointer; padding: .5rem .8rem; font-weight: 600; color: #57606a; user-select: none; }
.args[open] > summary { border-bottom: 1px solid #d0d7de; }
.args-table { width: 100%; border-collapse: collapse; font-size: .85rem; }
.args-table th, .args-table td { text-align: left; vertical-align: top; padding: .35rem .8rem; border-top: 1px solid #eaeef2; }
.args-table tr:first-child th, .args-table tr:first-child td { border-top: none; }
.args-table th { width: 30%; color: #57606a; font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-weight: 600; word-break: break-word; }
.args-table td { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; word-break: break-word; white-space: pre-wrap; }
.args-empty { margin: 0; padding: .5rem .8rem; color: #8c959f; }
.shots { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
.shot { min-width: 0; }
.shot .label { font-weight: 600; color: #57606a; margin: 0 0 .4rem; font-size: .9rem; }
.shot a { display: block; }
.shot img { width: 100%; height: auto; border: 1px solid #d0d7de; border-radius: 6px; background: #fff; }
.shot .missing { color: #8c959f; font-style: italic; }
@media (max-width: 720px) { .shots { grid-template-columns: 1fr; } }
""".strip()

_RECORDING_SCRIPT = """
(function () {
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.appendChild(area);
      area.select();
      try {
        if (!document.execCommand("copy")) {
          reject(new Error("copy failed"));
          return;
        }
        resolve();
      } catch (err) {
        reject(err);
      } finally {
        document.body.removeChild(area);
      }
    });
  }

  function flashCopied(btn) {
    var previous = btn.textContent;
    btn.textContent = "已複製";
    btn.classList.add("copied");
    window.setTimeout(function () {
      btn.textContent = previous;
      btn.classList.remove("copied");
    }, 1200);
  }

  function instructionCopyText(btn) {
    var instruction = btn.getAttribute("data-instruction") || "";
    if (!instruction) return "";
    var outcome = (btn.getAttribute("data-expected-outcome") || "").trim();
    if (!outcome) return instruction;
    return instruction + "\\n# expected_outcome: " + outcome;
  }

  Array.prototype.slice.call(document.querySelectorAll("button.copy-instruction")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var text = instructionCopyText(btn);
      if (!text) return;
      copyText(text).then(function () {
        flashCopied(btn);
      }).catch(function () {
        window.alert("無法複製指令，請手動選取文字。");
      });
    });
  });

  Array.prototype.slice.call(document.querySelectorAll("button.delete-instruction")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var group = btn.closest(".instruction-group");
      if (!group) return;
      if (window.location.protocol === "file:") {
        window.alert("無法刪除：請從主程式的「報告列表」開啟此頁（需本機服務）。");
        return;
      }
      var runId = group.getAttribute("data-run-id") || "";
      var eventIndex = group.getAttribute("data-event-index") || "";
      var titleEl = group.querySelector(".instruction-title");
      var label = titleEl ? (titleEl.textContent || "").trim() : "";
      if (!runId || !eventIndex) {
        window.alert("缺少事件資訊。");
        return;
      }
      var confirmText = label
        ? ("確定刪除指令「" + label + "」？\\n將無法復原。")
        : "確定刪除此筆指令？\\n將無法復原。";
      if (!window.confirm(confirmText)) return;
      btn.disabled = true;
      fetch("/api/runs/" + encodeURIComponent(runId) + "/events/" + encodeURIComponent(eventIndex) + "/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}"
      })
        .then(function (response) {
          return response.json().then(function (payload) {
            return { ok: response.ok, payload: payload };
          });
        })
        .then(function (result) {
          if (!result.ok || !result.payload || !result.payload.ok) {
            btn.disabled = false;
            var err = (result.payload && result.payload.error) || "刪除失敗";
            window.alert(err);
            return;
          }
          window.location.reload();
        })
        .catch(function () {
          btn.disabled = false;
          window.alert("無法連線主程式，請確認主程式正在執行。");
        });
    });
  });

  Array.prototype.slice.call(document.querySelectorAll("button.collapse-instruction")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var group = btn.closest(".instruction-group");
      if (!group) return;
      group.open = false;
      var summary = group.querySelector("summary");
      if (summary && typeof summary.scrollIntoView === "function") {
        summary.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });
  });

  var copyAll = document.querySelector("button.copy-all-instructions");
  if (copyAll) {
    copyAll.addEventListener("click", function () {
      var lines = Array.prototype.slice
        .call(document.querySelectorAll("button.copy-instruction[data-instruction]"))
        .map(function (btn) { return instructionCopyText(btn); })
        .filter(function (text) { return !!text; });
      if (!lines.length) return;
      copyText(lines.join("\\n")).then(function () {
        flashCopied(copyAll);
      }).catch(function () {
        window.alert("無法複製指令，請手動選取文字。");
      });
    });
  }

  var renameBtn = document.querySelector("button.rename-recording");
  if (renameBtn) {
    renameBtn.addEventListener("click", function () {
      if (window.location.protocol === "file:") {
        window.alert("無法重新命名：請從主程式的「報告列表」開啟此頁（需本機服務）。");
        return;
      }
      var runId = toolbarRunId();
      if (!runId) {
        window.alert("缺少錄製資訊。");
        return;
      }
      var nextName = window.prompt("重新命名錄製資料夾：", runId);
      if (nextName == null) return;
      nextName = String(nextName).trim();
      if (!nextName || nextName === runId) return;
      renameBtn.disabled = true;
      fetch("/api/runs/" + encodeURIComponent(runId) + "/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: nextName })
      })
        .then(function (response) {
          return response.json().then(function (payload) {
            return { ok: response.ok && payload && payload.ok, payload: payload || {} };
          }).catch(function () {
            return { ok: false, payload: {} };
          });
        })
        .then(function (result) {
          renameBtn.disabled = false;
          if (!result.ok) {
            window.alert((result.payload && result.payload.error) || "重新命名失敗。");
            return;
          }
          var newId = (result.payload && result.payload.new_id) || nextName;
          window.location.href = "../" + encodeURIComponent(newId) + "/recording_steps.html";
        })
        .catch(function () {
          renameBtn.disabled = false;
          window.alert("無法連線主程式，請確認主程式正在執行。");
        });
    });
  }

  function setLandmarksStatus(panel, text, isError) {
    var status = panel.querySelector(".landmarks-status");
    if (!status) return;
    status.textContent = text || "";
    if (isError) status.classList.add("error");
    else status.classList.remove("error");
  }

  function selectedHints(panel, group) {
    return Array.prototype.slice
      .call(panel.querySelectorAll('input[data-landmark-group="' + group + '"]:checked'))
      .map(function (input) {
        var side = input.getAttribute("data-side");
        return {
          label: input.getAttribute("data-label") || "",
          side: side === "" || side == null ? null : side
        };
      })
      .filter(function (item) { return !!item.label; });
  }

  function selectedPrimaryIndex(panel, group) {
    var checked = panel.querySelector(
      'input[type="radio"][data-primary-group="' + group + '"]:checked'
    );
    if (!checked) return null;
    var raw = checked.getAttribute("data-primary-index");
    if (raw == null || raw === "") return null;
    var value = parseInt(raw, 10);
    return Number.isFinite(value) ? value : null;
  }

  Array.prototype.slice.call(document.querySelectorAll("button.apply-landmarks")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var panel = btn.closest(".landmarks");
      var group = btn.closest(".instruction-group");
      if (!panel || !group) return;
      if (window.location.protocol === "file:") {
        setLandmarksStatus(panel, "請透過主程式開啟報告以套用地標。", true);
        return;
      }
      var runId = group.getAttribute("data-run-id") || "";
      var eventIndex = group.getAttribute("data-event-index") || "";
      var kind = group.getAttribute("data-kind") || "";
      if (!runId || !eventIndex) {
        setLandmarksStatus(panel, "缺少事件資訊。", true);
        return;
      }
      var body = { selected: selectedHints(panel, "start") };
      if (kind === "drag") {
        body.selected_end = selectedHints(panel, "end");
      }
      var primaryIndex = selectedPrimaryIndex(panel, "start");
      if (primaryIndex != null) body.primary_index = primaryIndex;
      if (kind === "drag") {
        var primaryEndIndex = selectedPrimaryIndex(panel, "end");
        if (primaryEndIndex != null) body.primary_end_index = primaryEndIndex;
      }
      btn.disabled = true;
      setLandmarksStatus(panel, "套用中…", false);
      fetch("/api/runs/" + encodeURIComponent(runId) + "/events/" + encodeURIComponent(eventIndex) + "/landmarks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      })
        .then(function (response) {
          return response.json().then(function (payload) {
            return { ok: response.ok, payload: payload };
          });
        })
        .then(function (result) {
          btn.disabled = false;
          if (!result.ok || !result.payload || !result.payload.ok) {
            var err = (result.payload && result.payload.error) || "套用失敗";
            setLandmarksStatus(panel, err, true);
            return;
          }
          if (result.payload.rebuilt) {
            window.location.reload();
            return;
          }
          var instruction = result.payload.instruction || "";
          var title = group.querySelector(".instruction-title");
          if (title) title.textContent = instruction;
          var copyBtn = group.querySelector("button.copy-instruction");
          if (copyBtn) copyBtn.setAttribute("data-instruction", instruction);
          btn.classList.add("applied");
          setLandmarksStatus(panel, "已套用", false);
          window.setTimeout(function () {
            btn.classList.remove("applied");
          }, 1200);
        })
        .catch(function () {
          btn.disabled = false;
          setLandmarksStatus(panel, "無法連線主程式，請確認主程式正在執行。", true);
        });
    });
  });

  Array.prototype.slice.call(document.querySelectorAll("button.rerun-yolo-ocr")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var panel = btn.closest(".vision-retry");
      var group = btn.closest(".instruction-group");
      if (!panel || !group) return;
      var status = panel.querySelector(".vision-retry-status");
      function setStatus(text, isError) {
        if (!status) return;
        status.textContent = text || "";
        if (isError) status.classList.add("error");
        else status.classList.remove("error");
      }
      if (window.location.protocol === "file:") {
        setStatus("請透過主程式開啟報告以重新偵測。", true);
        return;
      }
      var runId = group.getAttribute("data-run-id") || "";
      var eventIndex = group.getAttribute("data-event-index") || "";
      if (!runId || !eventIndex) {
        setStatus("缺少事件資訊。", true);
        return;
      }
      btn.disabled = true;
      setStatus("偵測中…可能需要數十秒。", false);
      fetch("/api/runs/" + encodeURIComponent(runId) + "/events/" + encodeURIComponent(eventIndex) + "/yolo_ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}"
      })
        .then(function (response) {
          return response.json().then(function (payload) {
            return { ok: response.ok, payload: payload };
          });
        })
        .then(function (result) {
          if (!result.ok || !result.payload || !result.payload.ok) {
            btn.disabled = false;
            var err = (result.payload && result.payload.error) || "重新偵測失敗";
            setStatus(err, true);
            return;
          }
          window.location.reload();
        })
        .catch(function () {
          btn.disabled = false;
          setStatus("無法連線主程式，請確認主程式正在執行。", true);
        });
    });
  });

  function setTypedTextStatus(panel, text, isError) {
    var status = panel.querySelector(".typed-text-status");
    if (!status) return;
    status.textContent = text || "";
    if (isError) status.classList.add("error");
    else status.classList.remove("error");
  }

  Array.prototype.slice.call(document.querySelectorAll("button.apply-typed-text")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      applyTypedText(btn);
    });
  });

  function syncTypedTextChoiceSelection(panel, text) {
    var matched = false;
    Array.prototype.slice.call(panel.querySelectorAll("button.typed-text-choice")).forEach(function (choice) {
      var choiceText = choice.getAttribute("data-text") || "";
      var selected = choiceText === text;
      if (selected) matched = true;
      if (selected) choice.classList.add("selected");
      else choice.classList.remove("selected");
    });
    return matched;
  }

  Array.prototype.slice.call(document.querySelectorAll("button.typed-text-choice")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var panel = btn.closest(".typed-text");
      if (!panel) return;
      var input = panel.querySelector(".typed-text-input");
      if (!input) return;
      input.value = btn.getAttribute("data-text") || "";
      syncTypedTextChoiceSelection(panel, input.value);
      setTypedTextStatus(panel, "", false);
    });
  });

  Array.prototype.slice.call(document.querySelectorAll(".typed-text-input")).forEach(function (input) {
    input.addEventListener("keydown", function (event) {
      if (event.key !== "Enter") return;
      event.preventDefault();
      event.stopPropagation();
      var panel = input.closest(".typed-text");
      var btn = panel ? panel.querySelector("button.apply-typed-text") : null;
      if (btn) applyTypedText(btn);
    });
    input.addEventListener("input", function () {
      var panel = input.closest(".typed-text");
      if (!panel) return;
      syncTypedTextChoiceSelection(panel, input.value);
      setTypedTextStatus(panel, "", false);
    });
  });

  function applyTypedText(btn) {
      var panel = btn.closest(".typed-text");
      var group = btn.closest(".instruction-group");
      if (!panel || !group) return;
      var input = panel.querySelector(".typed-text-input");
      if (!input) return;
      if (window.location.protocol === "file:") {
        setTypedTextStatus(panel, "請透過主程式開啟報告以修改文字。", true);
        return;
      }
      var runId = group.getAttribute("data-run-id") || "";
      var eventIndex = group.getAttribute("data-event-index") || "";
      if (!runId || !eventIndex) {
        setTypedTextStatus(panel, "缺少事件資訊。", true);
        return;
      }
      var text = input.value || "";
      btn.disabled = true;
      setTypedTextStatus(panel, "套用中…", false);
      fetch("/api/runs/" + encodeURIComponent(runId) + "/events/" + encodeURIComponent(eventIndex) + "/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
      })
        .then(function (response) {
          return response.json().then(function (payload) {
            return { ok: response.ok, payload: payload };
          });
        })
        .then(function (result) {
          btn.disabled = false;
          if (!result.ok || !result.payload || !result.payload.ok) {
            var err = (result.payload && result.payload.error) || "套用失敗";
            setTypedTextStatus(panel, err, true);
            return;
          }
          var instruction = result.payload.instruction || "";
          var saved = result.payload.text || "";
          input.value = saved;
          syncTypedTextChoiceSelection(panel, saved);
          var title = group.querySelector(".instruction-title");
          if (title && instruction) title.textContent = instruction;
          var copyBtn = group.querySelector("button.copy-instruction");
          if (copyBtn && instruction) copyBtn.setAttribute("data-instruction", instruction);
          btn.classList.add("applied");
          setTypedTextStatus(panel, "已套用", false);
          window.setTimeout(function () {
            btn.classList.remove("applied");
          }, 1200);
        })
        .catch(function () {
          btn.disabled = false;
          setTypedTextStatus(panel, "無法連線主程式，請確認主程式正在執行。", true);
        });
  }

  function setExpectedOutcomeStatus(panel, text, isError) {
    var status = panel.querySelector(".expected-outcome-status");
    if (!status) return;
    status.textContent = text || "";
    if (isError) status.classList.add("error");
    else status.classList.remove("error");
  }

  Array.prototype.slice.call(document.querySelectorAll("button.apply-expected-outcome")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      applyExpectedOutcome(btn);
    });
  });

  Array.prototype.slice.call(document.querySelectorAll(".expected-outcome-input")).forEach(function (input) {
    input.addEventListener("keydown", function (event) {
      if (!(event.key === "Enter" && (event.ctrlKey || event.metaKey))) return;
      event.preventDefault();
      event.stopPropagation();
      var panel = input.closest(".expected-outcome");
      var btn = panel ? panel.querySelector("button.apply-expected-outcome") : null;
      if (btn) applyExpectedOutcome(btn);
    });
  });

  function applyExpectedOutcome(btn) {
      var panel = btn.closest(".expected-outcome");
      var group = btn.closest(".instruction-group");
      if (!panel || !group) return;
      var input = panel.querySelector(".expected-outcome-input");
      if (!input) return;
      if (window.location.protocol === "file:") {
        setExpectedOutcomeStatus(panel, "請透過主程式開啟報告以修改預期結果。", true);
        return;
      }
      var runId = group.getAttribute("data-run-id") || "";
      var eventIndex = group.getAttribute("data-event-index") || "";
      if (!runId || !eventIndex) {
        setExpectedOutcomeStatus(panel, "缺少事件資訊。", true);
        return;
      }
      var text = input.value || "";
      btn.disabled = true;
      setExpectedOutcomeStatus(panel, "套用中…", false);
      fetch("/api/runs/" + encodeURIComponent(runId) + "/events/" + encodeURIComponent(eventIndex) + "/expected_outcome", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_outcome: text })
      })
        .then(function (response) {
          return response.json().then(function (payload) {
            return { ok: response.ok, payload: payload };
          });
        })
        .then(function (result) {
          btn.disabled = false;
          if (!result.ok || !result.payload || !result.payload.ok) {
            var err = (result.payload && result.payload.error) || "套用失敗";
            setExpectedOutcomeStatus(panel, err, true);
            return;
          }
          var saved = result.payload.expected_outcome;
          if (saved == null) saved = "";
          input.value = saved;
          var copyBtn = group.querySelector("button.copy-instruction");
          if (copyBtn) {
            if (saved) copyBtn.setAttribute("data-expected-outcome", saved);
            else copyBtn.removeAttribute("data-expected-outcome");
          }
          btn.classList.add("applied");
          setExpectedOutcomeStatus(panel, "已套用", false);
          window.setTimeout(function () {
            btn.classList.remove("applied");
          }, 1200);
        })
        .catch(function () {
          btn.disabled = false;
          setExpectedOutcomeStatus(panel, "無法連線主程式，請確認主程式正在執行。", true);
        });
  }

  function setStepInstructionStatus(panel, text, isError) {
    var status = panel.querySelector(".step-instruction-status");
    if (!status) return;
    status.textContent = text || "";
    if (isError) status.classList.add("error");
    else status.classList.remove("error");
  }

  Array.prototype.slice.call(document.querySelectorAll("button.apply-step-instruction")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      applyStepInstruction(btn);
    });
  });

  Array.prototype.slice.call(document.querySelectorAll(".step-instruction-input")).forEach(function (input) {
    input.addEventListener("keydown", function (event) {
      if (!(event.key === "Enter" && (event.ctrlKey || event.metaKey))) return;
      event.preventDefault();
      event.stopPropagation();
      var panel = input.closest(".step-instruction");
      var btn = panel ? panel.querySelector("button.apply-step-instruction") : null;
      if (btn) applyStepInstruction(btn);
    });
  });

  function applyStepInstruction(btn) {
      var panel = btn.closest(".step-instruction");
      var group = btn.closest(".instruction-group");
      if (!panel || !group) return;
      var input = panel.querySelector(".step-instruction-input");
      if (!input) return;
      if (window.location.protocol === "file:") {
        setStepInstructionStatus(panel, "請透過主程式開啟報告以修改指令。", true);
        return;
      }
      var runId = group.getAttribute("data-run-id") || "";
      var eventIndex = group.getAttribute("data-event-index") || "";
      if (!runId || !eventIndex) {
        setStepInstructionStatus(panel, "缺少事件資訊。", true);
        return;
      }
      var text = input.value || "";
      btn.disabled = true;
      setStepInstructionStatus(panel, "套用中…", false);
      fetch("/api/runs/" + encodeURIComponent(runId) + "/events/" + encodeURIComponent(eventIndex) + "/instruction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instruction: text })
      })
        .then(function (response) {
          return response.json().then(function (payload) {
            return { ok: response.ok, payload: payload };
          });
        })
        .then(function (result) {
          btn.disabled = false;
          if (!result.ok || !result.payload || !result.payload.ok) {
            var err = (result.payload && result.payload.error) || "套用失敗";
            setStepInstructionStatus(panel, err, true);
            return;
          }
          var saved = result.payload.instruction || "";
          input.value = saved;
          var title = group.querySelector(".instruction-title");
          if (title && saved) title.textContent = saved;
          var copyBtn = group.querySelector("button.copy-instruction");
          if (copyBtn && saved) copyBtn.setAttribute("data-instruction", saved);
          btn.classList.add("applied");
          setStepInstructionStatus(panel, "已套用", false);
          window.setTimeout(function () {
            btn.classList.remove("applied");
          }, 1200);
        })
        .catch(function () {
          btn.disabled = false;
          setStepInstructionStatus(panel, "無法連線主程式，請確認主程式正在執行。", true);
        });
  }

  var addDialog = document.getElementById("add-step-dialog");
  var addForm = addDialog ? addDialog.querySelector("form") : null;
  var addStatus = addDialog ? addDialog.querySelector(".add-step-status") : null;
  var instructionDirty = false;

  function toolbarRunId() {
    var toolbar = document.querySelector(".recording-toolbar");
    return toolbar ? (toolbar.getAttribute("data-run-id") || "") : "";
  }

  function setAddStatus(text, isError) {
    if (!addStatus) return;
    addStatus.textContent = text || "";
    if (isError) addStatus.classList.add("error");
    else addStatus.classList.remove("error");
  }

  function fieldForKind(kind) {
    return addForm ? addForm.querySelector('[data-kind-field="' + kind + '"]') : null;
  }

  function suggestedInstruction(kind) {
    if (!addForm) return "";
    if (kind === "text_input") {
      var text = ((addForm.querySelector('[name="text"]') || {}).value || "").trim();
      return text ? ("輸入「" + text + "」") : "";
    }
    if (kind === "key_press") {
      var key = ((addForm.querySelector('[name="key"]') || {}).value || "").trim();
      return key ? ("按下 " + key + " 鍵") : "";
    }
    if (kind === "hotkey") {
      var combo = ((addForm.querySelector('[name="keys"]') || {}).value || "").trim();
      return combo ? ("按下 " + combo) : "";
    }
    if (kind === "scroll") {
      var delta = ((addForm.querySelector('[name="scroll_delta"]') || {}).value || "");
      if (delta === "1") return "向上捲動";
      if (delta === "-1") return "向下捲動";
      return "";
    }
    if (kind === "wait") {
      var seconds = ((addForm.querySelector('[name="duration_seconds"]') || {}).value || "").trim();
      return seconds ? ("等待 " + seconds + " 秒") : "";
    }
    return "";
  }

  function syncKindFields() {
    if (!addForm) return;
    var kind = (addForm.querySelector('[name="kind"]') || {}).value || "click";
    Array.prototype.slice.call(addForm.querySelectorAll("[data-kind-field]")).forEach(function (row) {
      var match = row.getAttribute("data-kind-field") === kind;
      row.hidden = !match;
      Array.prototype.slice.call(row.querySelectorAll("input, select, textarea")).forEach(function (el) {
        el.disabled = !match;
      });
    });
    if (!instructionDirty) {
      var instruction = addForm.querySelector('[name="instruction"]');
      if (instruction) instruction.value = suggestedInstruction(kind);
    }
  }

  function closeAddDialog() {
    if (!addDialog) return;
    addDialog.classList.remove("open");
    addDialog.setAttribute("hidden", "");
    setAddStatus("", false);
  }

  function openAddDialog(afterEventIndex) {
    if (!addDialog || !addForm) return;
    if (window.location.protocol === "file:") {
      window.alert("無法新增：請從主程式的「報告列表」開啟此頁（需本機服務）。");
      return;
    }
    addForm.reset();
    instructionDirty = false;
    addForm.setAttribute("data-after-event-index", afterEventIndex == null ? "" : String(afterEventIndex));
    var kindSelect = addForm.querySelector('[name="kind"]');
    if (kindSelect) kindSelect.value = "click";
    syncKindFields();
    setAddStatus("", false);
    addDialog.removeAttribute("hidden");
    addDialog.classList.add("open");
    var first = addForm.querySelector('[name="kind"]');
    if (first && typeof first.focus === "function") first.focus();
  }

  Array.prototype.slice.call(document.querySelectorAll("button.add-instruction")).forEach(function (btn) {
    btn.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      var group = btn.closest(".instruction-group");
      var eventIndex = group ? group.getAttribute("data-event-index") : "";
      var parsed = eventIndex ? parseInt(eventIndex, 10) : NaN;
      openAddDialog(Number.isFinite(parsed) ? parsed : null);
    });
  });

  var addToolbar = document.querySelector("button.add-recording-step");
  if (addToolbar) {
    addToolbar.addEventListener("click", function (event) {
      event.preventDefault();
      openAddDialog(null);
    });
  }

  if (addDialog) {
    addDialog.addEventListener("click", function (event) {
      if (event.target === addDialog) closeAddDialog();
    });
  }
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && addDialog && addDialog.classList.contains("open")) {
      closeAddDialog();
    }
  });
  if (addForm) {
    var kindSelect = addForm.querySelector('[name="kind"]');
    if (kindSelect) {
      kindSelect.addEventListener("change", function () {
        instructionDirty = false;
        syncKindFields();
      });
    }
    var instructionInput = addForm.querySelector('[name="instruction"]');
    if (instructionInput) {
      instructionInput.addEventListener("input", function () {
        instructionDirty = true;
      });
    }
    Array.prototype.slice.call(addForm.querySelectorAll("[data-kind-field] input, [data-kind-field] select")).forEach(function (el) {
      el.addEventListener("input", function () {
        if (!instructionDirty) syncKindFields();
      });
      el.addEventListener("change", function () {
        if (!instructionDirty) syncKindFields();
      });
    });
    var cancelBtn = addForm.querySelector("button.add-step-cancel");
    if (cancelBtn) {
      cancelBtn.addEventListener("click", function (event) {
        event.preventDefault();
        closeAddDialog();
      });
    }
    addForm.addEventListener("submit", function (event) {
      event.preventDefault();
      var runId = toolbarRunId();
      if (!runId) {
        setAddStatus("缺少錄製資訊。", true);
        return;
      }
      var kind = (addForm.querySelector('[name="kind"]') || {}).value || "";
      var instruction = ((addForm.querySelector('[name="instruction"]') || {}).value || "").trim();
      var expectedOutcome = ((addForm.querySelector('[name="expected_outcome"]') || {}).value || "");
      var afterRaw = addForm.getAttribute("data-after-event-index") || "";
      var body = { kind: kind, instruction: instruction, expected_outcome: expectedOutcome };
      if (afterRaw !== "") {
        var afterIndex = parseInt(afterRaw, 10);
        if (Number.isFinite(afterIndex)) body.after_event_index = afterIndex;
      }
      if (kind === "text_input") {
        body.text = ((addForm.querySelector('[name="text"]') || {}).value || "");
      } else if (kind === "key_press") {
        body.key = ((addForm.querySelector('[name="key"]') || {}).value || "");
      } else if (kind === "hotkey") {
        body.keys = ((addForm.querySelector('[name="keys"]') || {}).value || "");
      } else if (kind === "scroll") {
        var scrollRaw = ((addForm.querySelector('[name="scroll_delta"]') || {}).value || "");
        body.scroll_delta = parseInt(scrollRaw, 10);
      } else if (kind === "wait") {
        var durationRaw = ((addForm.querySelector('[name="duration_seconds"]') || {}).value || "");
        body.duration_seconds = Number(durationRaw);
      }
      var submitBtn = addForm.querySelector("button.add-step-submit");
      if (submitBtn) submitBtn.disabled = true;
      setAddStatus("新增中…", false);
      fetch("/api/runs/" + encodeURIComponent(runId) + "/events/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      })
        .then(function (response) {
          return response.json().then(function (payload) {
            return { ok: response.ok, payload: payload };
          });
        })
        .then(function (result) {
          if (submitBtn) submitBtn.disabled = false;
          if (!result.ok || !result.payload || !result.payload.ok) {
            var err = (result.payload && result.payload.error) || "新增失敗";
            setAddStatus(err, true);
            return;
          }
          var newIndex = result.payload.event_index;
          var hash = newIndex ? ("#event-" + newIndex) : "";
          window.location.hash = hash;
          window.location.reload();
        })
        .catch(function () {
          if (submitBtn) submitBtn.disabled = false;
          setAddStatus("無法連線主程式，請確認主程式正在執行。", true);
        });
    });
  }

  (function openHashedEvent() {
    var hash = window.location.hash || "";
    var match = hash.match(/^#event-(\\d+)$/);
    if (!match) return;
    var group = document.getElementById("event-" + match[1]);
    if (!group) return;
    group.open = true;
    if (typeof group.scrollIntoView === "function") {
      group.scrollIntoView({ block: "start" });
    }
  })();
})();
""".strip()

_INDEX_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Microsoft JhengHei", system-ui, sans-serif;
  margin: 0; padding: 2rem; line-height: 1.5;
  background: #f5f6f8; color: #1f2328;
}
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
.intro { color: #57606a; margin: 0 0 .75rem; }
.bulk-bar {
  display: flex; flex-wrap: wrap; align-items: center; gap: .5rem .75rem;
  margin: 0 0 1.25rem; padding: .65rem .9rem;
  background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
}
.bulk-bar .bulk-count { color: #57606a; font-size: .9rem; min-width: 5.5rem; }
.bulk-bar button {
  appearance: none; border: 1px solid #d0d7de; background: #f6f8fa;
  cursor: pointer; border-radius: 6px; padding: .35rem .7rem;
  font-size: .85rem; line-height: 1.2; font-family: inherit;
}
.bulk-bar button:hover:not(:disabled) { background: #eaeef2; }
.bulk-bar button:disabled { opacity: .45; cursor: not-allowed; }
.bulk-bar .bulk-bug { color: #9a6700; border-color: #d4a72c; background: #fff8c5; }
.bulk-bar .bulk-bug:hover:not(:disabled) { background: #fff1a8; }
.bulk-bar .bulk-delete { color: #cf222e; border-color: #ff8182; background: #ffebe9; }
.bulk-bar .bulk-delete:hover:not(:disabled) { background: #ffd7d5; }
.empty { color: #8c959f; font-style: italic; }
.reports {
  width: 100%; border-collapse: collapse; background: #fff;
  border: 1px solid #d0d7de; border-radius: 10px; overflow: hidden;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.reports th, .reports td {
  text-align: left; vertical-align: top; padding: .75rem 1rem;
  border-top: 1px solid #d0d7de;
}
.reports thead th {
  border-top: none; background: #f6f8fa; color: #57606a; font-size: .85rem;
}
.reports thead th.sortable {
  cursor: pointer; user-select: none; white-space: nowrap;
}
.reports thead th.sortable:hover { color: #1f2328; background: #eaeef2; }
.reports thead th.sortable::after {
  content: "⇅"; display: inline-block; margin-left: .35rem;
  font-size: .7rem; opacity: .35; vertical-align: middle;
}
.reports thead th.sortable[aria-sort="ascending"]::after { content: "▲"; opacity: .85; }
.reports thead th.sortable[aria-sort="descending"]::after { content: "▼"; opacity: .85; }
.reports thead th.no-sort { width: 5.5rem; text-align: center; }
.reports thead th.select-col,
.reports td.select-col {
  width: 2.5rem; text-align: center; vertical-align: middle; padding-left: .75rem; padding-right: .5rem;
}
.reports td.actions { text-align: center; vertical-align: middle; white-space: nowrap; }
.select-run, .select-all {
  appearance: none; -webkit-appearance: none;
  width: 1rem; height: 1rem; cursor: pointer;
  margin: 0; vertical-align: middle;
  border: 1.5px solid #c9d1d9; border-radius: 3px;
  background: #f6f8fa; box-shadow: inset 0 0 0 1px #fff;
}
.select-run:hover, .select-all:hover { border-color: #96c8ff; background: #eef5ff; }
.select-run:checked, .select-all:checked {
  border-color: #54aeff; background: #54aeff;
  box-shadow: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round' d='M3.5 8.5l3 3 6-6'/%3E%3C/svg%3E");
  background-size: 100% 100%;
}
.select-run:indeterminate, .select-all:indeterminate {
  border-color: #54aeff; background: #54aeff;
  box-shadow: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Cpath fill='none' stroke='%23ffffff' stroke-width='2.2' stroke-linecap='round' d='M4 8h8'/%3E%3C/svg%3E");
  background-size: 100% 100%;
}
.select-run:disabled, .select-all:disabled { opacity: .45; cursor: wait; }
.delete-run, .bug-run {
  appearance: none; border: 1px solid transparent; background: transparent;
  cursor: pointer; border-radius: 6px;
  padding: .2rem .4rem; font-size: 1rem; line-height: 1;
}
.delete-run { color: #cf222e; }
.delete-run:hover { background: #ffebe9; border-color: #ff8182; }
.delete-run:disabled { opacity: .45; cursor: wait; }
.bug-run { color: #9a6700; }
.bug-run:hover { background: #fff8c5; border-color: #d4a72c; }
.bug-run:disabled { opacity: .45; cursor: wait; }
.reports tbody tr:hover { background: #f6f8fa; }
.reports tbody tr.selected { background: #ddf4ff; }
.reports tbody tr.selected:hover { background: #cceafc; }
.reports a { color: #0969da; text-decoration: none; font-weight: 600; word-break: break-all; }
.reports a:hover { text-decoration: underline; }
.badge { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }
.badge.ok { background: #dafbe1; color: #116329; }
.badge.fail { background: #ffebe9; color: #cf222e; }
.badge.neutral { background: #eaeef2; color: #57606a; }
.mono { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: .9rem; }
.tabs {
  display: flex; flex-wrap: wrap; gap: .35rem; margin: 0 0 1.25rem;
  border-bottom: 1px solid #d0d7de; padding-bottom: .35rem;
}
.tabs button {
  appearance: none; border: 1px solid transparent; background: transparent;
  cursor: pointer; border-radius: 6px 6px 0 0; padding: .45rem .9rem;
  font-size: .95rem; font-family: inherit; color: #57606a; font-weight: 600;
}
.tabs button:hover { color: #1f2328; background: #eaeef2; }
.tabs button.active {
  color: #0969da; border-color: #d0d7de #d0d7de #f5f6f8; background: #fff;
  margin-bottom: -1px; border-bottom-color: #fff;
}
.tab-panel[hidden] { display: none; }
@media (max-width: 720px) {
  .instruction-edit-panels { grid-template-columns: 1fr; }
  .reports, .reports thead, .reports tbody, .reports th, .reports td, .reports tr { display: block; }
  .reports thead { display: none; }
  .reports tr { border-top: 1px solid #d0d7de; padding: .5rem 0; }
  .reports td { border-top: none; padding: .25rem 1rem; }
  .reports td.select-col { padding-top: .5rem; }
  .reports td.select-col::before { display: none; }
  .reports td::before {
    content: attr(data-label); display: block; color: #57606a;
    font-size: .75rem; font-weight: 600; margin-bottom: .1rem;
  }
}
""".strip()

_INDEX_SCRIPT = """
(function () {
  function requireLocalServer() {
    if (!window.location.protocol || window.location.protocol === "file:") {
      window.alert("無法操作：請從主程式的「報告列表」開啟此頁（需本機服務）。");
      return false;
    }
    return true;
  }

  function postRunAction(runId, action) {
    return fetch("/api/runs/" + encodeURIComponent(runId) + "/" + action, { method: "POST" })
      .then(function (response) {
        return response.json().then(function (payload) {
          return { ok: response.ok && payload && payload.ok, payload: payload || {}, status: response.status };
        }).catch(function () {
          return { ok: false, payload: {}, status: response.status };
        });
      });
  }

  function formatSelectionSummary(items) {
    var labels = items.map(function (item) { return item.label; });
    var preview = labels.slice(0, 8).join("、");
    if (labels.length > 8) preview += "…（共 " + labels.length + " 筆）";
    return preview;
  }

  function initTabs() {
    var buttons = Array.prototype.slice.call(document.querySelectorAll(".tabs button[data-tab]"));
    var panels = Array.prototype.slice.call(document.querySelectorAll(".tab-panel[data-tab]"));
    if (!buttons.length || !panels.length) return;
    var allowed = { runs: true, smart: true, recordings: true };

    function activate(tabId) {
      var resolved = allowed[tabId] ? tabId : "runs";
      buttons.forEach(function (btn) {
        var active = btn.getAttribute("data-tab") === resolved;
        btn.classList.toggle("active", active);
        btn.setAttribute("aria-selected", active ? "true" : "false");
      });
      panels.forEach(function (panel) {
        var active = panel.getAttribute("data-tab") === resolved;
        if (active) panel.removeAttribute("hidden");
        else panel.setAttribute("hidden", "");
      });
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, "", "#" + resolved);
      } else {
        window.location.hash = resolved;
      }
    }

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        activate(btn.getAttribute("data-tab") || "runs");
      });
    });

    var hash = (window.location.hash || "").replace(/^#/, "");
    activate(allowed[hash] ? hash : "runs");
  }

  function initPanel(panel) {
    var table = panel.querySelector("table.reports");
    if (!table || !table.tBodies.length) return;
    var tbody = table.tBodies[0];
    var headers = Array.prototype.slice.call(table.querySelectorAll("thead th"));
    var sortCol = -1;
    var sortAsc = true;
    var selectAll = panel.querySelector("input.select-all");
    var bulkCount = panel.querySelector(".bulk-count");
    var bulkBug = panel.querySelector("button.bulk-bug");
    var bulkDelete = panel.querySelector("button.bulk-delete");
    var intro = panel.querySelector("p.intro");
    var kind = panel.getAttribute("data-tab") || "runs";
    var bulkBusy = false;

    function cellValue(row, col) {
      var cell = row.cells[col];
      if (!cell) return "";
      if (cell.getAttribute("data-sort") != null) return cell.getAttribute("data-sort");
      return (cell.textContent || "").trim();
    }

    function isEmpty(value) {
      return value === "" || value === "—";
    }

    function compareRows(a, b, type) {
      var va = cellValue(a, sortCol);
      var vb = cellValue(b, sortCol);
      if (isEmpty(va) && isEmpty(vb)) return 0;
      if (isEmpty(va)) return 1;
      if (isEmpty(vb)) return -1;
      var cmp;
      if (type === "num") {
        cmp = parseFloat(va) - parseFloat(vb);
        if (isNaN(cmp)) cmp = 0;
      } else {
        cmp = va.localeCompare(vb, undefined, { numeric: true, sensitivity: "base" });
      }
      return sortAsc ? cmp : -cmp;
    }

    function applySort(col, ascending) {
      var th = headers[col];
      if (!th || th.classList.contains("no-sort")) return;
      var type = th.getAttribute("data-type") || "text";
      sortCol = col;
      sortAsc = ascending;
      headers.forEach(function (header) {
        header.removeAttribute("aria-sort");
      });
      th.setAttribute("aria-sort", sortAsc ? "ascending" : "descending");
      var rows = Array.prototype.slice.call(tbody.rows);
      rows.sort(function (a, b) { return compareRows(a, b, type); });
      rows.forEach(function (row) { tbody.appendChild(row); });
    }

    function isTimeColumn(th) {
      return (th.textContent || "").trim() === "時間";
    }

    headers.forEach(function (th, col) {
      if (th.classList.contains("no-sort")) return;
      th.classList.add("sortable");
      th.setAttribute("tabindex", "0");
      th.setAttribute("role", "columnheader");
      th.setAttribute("title", "點選排序");

      function sortByColumn() {
        if (sortCol === col) {
          applySort(col, !sortAsc);
        } else {
          // 時間：預設新→舊；其他欄位：預設升冪
          applySort(col, !isTimeColumn(th));
        }
      }

      th.addEventListener("click", sortByColumn);
      th.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          sortByColumn();
        }
      });
    });

    // 預設依時間由新到舊
    var defaultTimeCol = -1;
    headers.forEach(function (th, col) {
      if (!th.classList.contains("no-sort") && isTimeColumn(th)) {
        defaultTimeCol = col;
      }
    });
    if (defaultTimeCol >= 0) {
      applySort(defaultTimeCol, false);
    }

    function introHelpText(count) {
      if (kind === "recordings") {
        return "共 " + count + " 筆錄製。勾選多筆後可批次回報或刪除；點選錄製名稱開啟事件紀錄；點選欄位標題可排序；🐛 可回報 bug；垃圾桶可刪除整份錄製資料夾。";
      }
      if (kind === "smart") {
        return "共 " + count + " 筆智能模式報告。勾選多筆後可批次回報或刪除；點選目標開啟步驟紀錄；點選欄位標題可排序；🐛 可回報 bug；垃圾桶可刪除整份報告資料夾。";
      }
      return "共 " + count + " 筆報告。勾選多筆後可批次回報或刪除；點選執行名稱開啟步驟紀錄；點選欄位標題可排序；🐛 可回報 bug；垃圾桶可刪除整份報告資料夾。";
    }

    function updateIntroCount() {
      if (!intro) return;
      intro.textContent = introHelpText(tbody.rows.length);
    }

    function selectedCheckboxes() {
      return Array.prototype.slice.call(tbody.querySelectorAll("input.select-run:checked"));
    }

    function syncRowSelected(checkbox) {
      var row = checkbox.closest("tr");
      if (!row) return;
      if (checkbox.checked) row.classList.add("selected");
      else row.classList.remove("selected");
    }

    function updateBulkBar() {
      var selected = selectedCheckboxes();
      var count = selected.length;
      var total = tbody.querySelectorAll("input.select-run").length;
      if (bulkCount) bulkCount.textContent = "已選 " + count + " 筆";
      if (bulkBug) bulkBug.disabled = bulkBusy || count === 0;
      if (bulkDelete) bulkDelete.disabled = bulkBusy || count === 0;
      if (selectAll) {
        selectAll.checked = total > 0 && count === total;
        selectAll.indeterminate = count > 0 && count < total;
        selectAll.disabled = bulkBusy || total === 0;
      }
    }

    function setBulkBusy(busy) {
      bulkBusy = busy;
      Array.prototype.slice.call(tbody.querySelectorAll("input.select-run")).forEach(function (cb) {
        cb.disabled = busy;
      });
      updateBulkBar();
    }

    function getSelectedItems() {
      return selectedCheckboxes().map(function (cb) {
        return {
          checkbox: cb,
          runId: cb.getAttribute("data-run-id") || "",
          label: cb.getAttribute("data-run-label") || cb.getAttribute("data-run-id") || "",
          row: cb.closest("tr"),
        };
      }).filter(function (item) { return !!item.runId; });
    }

    Array.prototype.slice.call(tbody.querySelectorAll("input.select-run")).forEach(function (cb) {
      syncRowSelected(cb);
      cb.addEventListener("change", function () {
        syncRowSelected(cb);
        updateBulkBar();
      });
      cb.addEventListener("click", function (event) {
        event.stopPropagation();
      });
    });

    if (selectAll) {
      selectAll.addEventListener("click", function (event) {
        event.stopPropagation();
      });
      selectAll.addEventListener("change", function () {
        var checked = selectAll.checked;
        Array.prototype.slice.call(tbody.querySelectorAll("input.select-run")).forEach(function (cb) {
          if (cb.disabled) return;
          cb.checked = checked;
          syncRowSelected(cb);
        });
        updateBulkBar();
      });
    }

    if (bulkBug) {
      bulkBug.addEventListener("click", function (event) {
        event.preventDefault();
        var items = getSelectedItems();
        if (!items.length) return;
        var summary = formatSelectionSummary(items);
        if (!window.confirm("確定回報選取的 " + items.length + " 筆報告？\\n" + summary + "\\n將壓縮各執行資料夾並複製到 \\\\\\\\192.168.0.9\\\\Joseph\\\\CUA-BUG。")) {
          return;
        }
        if (!requireLocalServer()) return;
        setBulkBusy(true);
        var okCount = 0;
        var failures = [];
        var chain = Promise.resolve();
        items.forEach(function (item) {
          chain = chain.then(function () {
            return postRunAction(item.runId, "bug").then(function (result) {
              if (result.ok) {
                okCount += 1;
              } else {
                var message = (result.payload && result.payload.error) || ("HTTP " + result.status);
                failures.push(item.label + "：" + message);
              }
            }).catch(function () {
              failures.push(item.label + "：無法連線本機服務");
            });
          });
        });
        chain.then(function () {
          setBulkBusy(false);
          var lines = ["成功 " + okCount + " / 失敗 " + failures.length];
          if (failures.length) lines = lines.concat(failures.slice(0, 10));
          if (failures.length > 10) lines.push("…其餘 " + (failures.length - 10) + " 筆略");
          window.alert(lines.join("\\n"));
        });
      });
    }

    if (bulkDelete) {
      bulkDelete.addEventListener("click", function (event) {
        event.preventDefault();
        var items = getSelectedItems();
        if (!items.length) return;
        var summary = formatSelectionSummary(items);
        if (!window.confirm("確定刪除選取的 " + items.length + " 筆報告？\\n" + summary + "\\n將刪除整個資料夾，且無法復原。")) {
          return;
        }
        if (!requireLocalServer()) return;
        setBulkBusy(true);
        var failures = [];
        var chain = Promise.resolve();
        items.forEach(function (item) {
          chain = chain.then(function () {
            return postRunAction(item.runId, "delete").then(function (result) {
              if (result.ok) {
                if (item.row && item.row.parentNode) item.row.parentNode.removeChild(item.row);
              } else {
                var message = (result.payload && result.payload.error) || ("HTTP " + result.status);
                failures.push(item.label + "：" + message);
              }
            }).catch(function () {
              failures.push(item.label + "：無法連線本機服務");
            });
          });
        });
        chain.then(function () {
          if (!tbody.rows.length) {
            window.location.reload();
            return;
          }
          setBulkBusy(false);
          updateIntroCount();
          updateBulkBar();
          if (failures.length) {
            var lines = ["部分刪除失敗（" + failures.length + " 筆）："].concat(failures.slice(0, 10));
            if (failures.length > 10) lines.push("…其餘 " + (failures.length - 10) + " 筆略");
            window.alert(lines.join("\\n"));
          }
        });
      });
    }

    Array.prototype.slice.call(panel.querySelectorAll("button.bug-run")).forEach(function (btn) {
      btn.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        var runId = btn.getAttribute("data-run-id") || "";
        if (!runId) return;
        var label = btn.getAttribute("data-run-label") || runId;
        if (!window.confirm("確定回報「" + label + "」？\\n將壓縮該執行資料夾並複製到 \\\\\\\\192.168.0.9\\\\Joseph\\\\CUA-BUG。")) {
          return;
        }
        if (!requireLocalServer()) return;
        btn.disabled = true;
        postRunAction(runId, "bug")
          .then(function (result) {
            btn.disabled = false;
            if (!result.ok) {
              var message = (result.payload && result.payload.error) || ("HTTP " + result.status);
              window.alert("回報失敗：" + message);
              return;
            }
            var copied = (result.payload && result.payload.copied_to) || "";
            window.alert(copied ? ("已複製到：\\n" + copied) : "已壓縮並複製到 bug 分享資料夾。");
          })
          .catch(function () {
            window.alert("無法回報：請從主程式的「報告列表」開啟此頁（需本機服務）。");
            btn.disabled = false;
          });
      });
    });

    Array.prototype.slice.call(panel.querySelectorAll("button.delete-run")).forEach(function (btn) {
      btn.addEventListener("click", function (event) {
        event.preventDefault();
        event.stopPropagation();
        var runId = btn.getAttribute("data-run-id") || "";
        if (!runId) return;
        var label = btn.getAttribute("data-run-label") || runId;
        if (!window.confirm("確定刪除報告「" + label + "」？\\n將刪除整個資料夾，且無法復原。")) {
          return;
        }
        if (!requireLocalServer()) return;
        btn.disabled = true;
        postRunAction(runId, "delete")
          .then(function (result) {
            if (!result.ok) {
              var message = (result.payload && result.payload.error) || ("HTTP " + result.status);
              window.alert("刪除失敗：" + message);
              btn.disabled = false;
              return;
            }
            var row = btn.closest("tr");
            if (row && row.parentNode) row.parentNode.removeChild(row);
            if (!tbody.rows.length) {
              window.location.reload();
              return;
            }
            updateIntroCount();
            updateBulkBar();
          })
          .catch(function () {
            window.alert("無法刪除：請從主程式的「報告列表」開啟此頁（需本機服務）。");
            btn.disabled = false;
          });
      });
    });

    updateBulkBar();
  }

  initTabs();
  Array.prototype.slice.call(document.querySelectorAll(".tab-panel")).forEach(initPanel);
})();
""".strip()


def session_html_path(run_root: Path) -> Path:
    return run_root / _HTML_NAME


def recording_html_path(run_root: Path) -> Path:
    return run_root / _RECORDING_HTML_NAME


def runs_index_html_path(runs_root: Path) -> Path:
    return Path(runs_root) / _INDEX_HTML_NAME


def _hand_csv_path(run_root: Path) -> Path:
    return run_root / "hand.csv"


def _coerce_args(value: Any) -> Any:
    """Parse an args value that may be a JSON string, a Python ``repr`` dict, or already structured.

    ``hand.csv`` stores args via ``csv.DictWriter`` (Python repr with single quotes), so plain
    ``json.loads`` fails. Try JSON first, then ``ast.literal_eval``.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except (ValueError, json.JSONDecodeError):
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _flatten_args_pairs(args: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten nested args into ``(key, value)`` pairs for table rendering."""
    pairs: list[tuple[str, str]] = []

    if isinstance(args, dict):
        for key, value in args.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                pairs.extend(_flatten_args_pairs(value, full_key))
            elif isinstance(value, list):
                if not value:
                    pairs.append((full_key, "（空）"))
                else:
                    for index, item in enumerate(value):
                        item_key = f"{full_key}[{index}]"
                        if isinstance(item, (dict, list)):
                            pairs.extend(_flatten_args_pairs(item, item_key))
                        else:
                            pairs.append((item_key, str(item)))
            else:
                pairs.append((full_key, str(value)))
        return pairs

    if isinstance(args, list):
        for index, item in enumerate(args):
            item_key = f"{prefix}[{index}]" if prefix else f"[{index}]"
            if isinstance(item, (dict, list)):
                pairs.extend(_flatten_args_pairs(item, item_key))
            else:
                pairs.append((item_key, str(item)))
        return pairs

    if args is None:
        return []
    return [("值", str(args))]


_INSTRUCTION_ARG_KEYS = frozenset(
    {"instruction", "start_instruction", "destination_instruction"}
)
_INSTRUCTION_ARG_LABELS = {
    "instruction": "指令",
    "start_instruction": "起點指令",
    "destination_instruction": "終點指令",
}


def _instruction_meta_pairs(args: Any) -> list[tuple[str, str]]:
    """Return ``(label, value)`` pairs for instruction fields stored in tool args."""
    if not isinstance(args, dict):
        return []

    pairs: list[tuple[str, str]] = []
    for key in ("instruction", "start_instruction", "destination_instruction"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            pairs.append((_INSTRUCTION_ARG_LABELS[key], value.strip()))
    return pairs


def _args_without_instructions(args: Any) -> Any:
    if not isinstance(args, dict):
        return args
    return {key: value for key, value in args.items() if key not in _INSTRUCTION_ARG_KEYS}


def _render_instruction_meta_html(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    return "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>" for label, value in pairs
    )


def _render_args_html(args: dict[str, Any] | Any) -> str:
    """Render args as a key/value table inside a collapsed ``<details>`` block."""
    pairs = _flatten_args_pairs(args)
    if pairs:
        rows = "".join(
            f"<tr><th>{escape(key)}</th><td>{escape(value)}</td></tr>" for key, value in pairs
        )
        body = f'<table class="args-table"><tbody>{rows}</tbody></table>'
    else:
        body = '<p class="args-empty">（無）</p>'
    return f'<details class="args"><summary>參數</summary>{body}</details>'


def _timestamp_text(timestamp: datetime | str | None) -> str:
    """Format a timestamp for display, e.g. ``2026-07-17 14:30:32 (UTC+08:00)``.

    Parses ISO 8601 input, converts timezone-aware values to local time, and drops microseconds.
    Falls back to the raw string when it cannot be parsed.
    """
    if timestamp is None:
        return ""
    if isinstance(timestamp, datetime):
        dt: datetime | None = timestamp
    else:
        text = str(timestamp).strip()
        if not text:
            return ""
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text

    if dt.tzinfo is not None:
        dt = dt.astimezone()
        offset = dt.strftime("%z")
        formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
        if offset:
            return f"{formatted} (UTC{offset[:3]}:{offset[3:]})"
        return formatted
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _status_text(ok: bool) -> str:
    return "成功" if ok else "失敗"


def _parse_csv_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


def _resolve_run_screenshot(raw: str | None, run_root: Path) -> Path | None:
    """Resolve a ``hand.csv`` screenshot path to an existing file, regardless of cwd.

    Stored paths may be absolute or relative to the repo root (the app's working directory),
    e.g. ``runs/<run_id>/eye/<file>.png``. Returns ``None`` when no candidate exists.
    """
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    repo_root = run_root.parent.parent
    for resolved in (repo_root / candidate, run_root / "eye" / candidate.name):
        if resolved.is_file():
            return resolved
    return None


def _relative_img_src(screenshot: Path | None, run_root: Path) -> str | None:
    """Return an ``<img src>`` value relative to the HTML file (which lives in ``run_root``)."""
    if screenshot is None:
        return None
    resolved = screenshot.resolve()
    try:
        rel = resolved.relative_to(run_root.resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        # Screenshot lives outside the run folder; fall back to an absolute file URI.
        return resolved.as_uri()


def _render_shot_html(label: str, screenshot: Path | None, run_root: Path) -> str:
    src = _relative_img_src(screenshot, run_root)
    if src is None:
        body = '<p class="missing">無螢幕截圖</p>'
    else:
        esc = escape(src, quote=True)
        body = f'<a href="{esc}" target="_blank" rel="noopener"><img src="{esc}" alt="{escape(label)}" loading="lazy"></a>'
    return f'<div class="shot"><p class="label">{escape(label)}</p>{body}</div>'


def _normalize_timestamp_key(timestamp: datetime | str | None) -> str | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        return timestamp.isoformat()
    text = str(timestamp).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()
    except ValueError:
        return text


def _load_session_report_data(run_root: Path) -> dict[str, Any]:
    report = _load_run_report(run_root)
    if report is not None:
        return report
    from src.common.session_report import build_session_report

    return build_session_report(run_root, session_end_reason="")


def _iter_hand_csv_rows(run_root: Path) -> list[dict[str, str]]:
    hand_csv = _hand_csv_path(run_root)
    if not hand_csv.is_file() or hand_csv.stat().st_size == 0:
        return []

    rows: list[dict[str, str]] = []
    with hand_csv.open(newline="", encoding="utf-8") as handle:
        first_line = handle.readline()
        handle.seek(0)
        has_header = first_line.split(",", 1)[0].strip() == "timestamp"
        reader = (
            csv.DictReader(handle)
            if has_header
            else csv.DictReader(handle, fieldnames=_HAND_CSV_FIELDS)
        )
        for row in reader:
            rows.append({key: (value or "") for key, value in row.items()})
    return rows


def _hand_operation_from_row(
    *,
    run_root: Path,
    operation_number: int,
    row: dict[str, str],
) -> dict[str, Any]:
    before = _resolve_run_screenshot(
        row.get("screenshot_before_path") or row.get("screenshot_name"),
        run_root,
    )
    after = _resolve_run_screenshot(row.get("screenshot_after_path"), run_root)
    return {
        "operation_number": operation_number,
        "action": row.get("action") or "",
        "args": _coerce_args(row.get("args")),
        "ok": _parse_csv_bool(row.get("ok")),
        "message": row.get("message") or "",
        "timestamp": row.get("timestamp") or None,
        "before": before,
        "after": after,
    }


_MISSING = object()


def _step_verify_meta_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract expected outcome / verify / status from a step JSON or report step record."""
    meta: dict[str, Any] = {}
    step_timing = payload.get("step_timing")
    timing_source = step_timing if isinstance(step_timing, dict) else {}

    if "expected_outcome" in timing_source:
        expected_outcome = timing_source.get("expected_outcome")
    elif "expected_outcome" in payload:
        expected_outcome = payload.get("expected_outcome")
    else:
        expected_outcome = _MISSING
    if expected_outcome is not _MISSING:
        if isinstance(expected_outcome, str):
            meta["expected_outcome"] = expected_outcome.strip()
        else:
            meta["expected_outcome"] = None

    if "verify" in timing_source:
        verify = timing_source.get("verify")
    elif "verify" in payload:
        verify = payload.get("verify")
    else:
        verify = _MISSING
    if verify is not _MISSING:
        meta["verify"] = verify if isinstance(verify, dict) else None

    status = None
    if isinstance(timing_source.get("status"), str):
        status = timing_source.get("status")
    timing = payload.get("timing")
    if status is None and isinstance(timing, dict) and isinstance(timing.get("status"), str):
        status = timing.get("status")
    if isinstance(status, str) and status.strip():
        meta["status"] = status.strip()
    return meta


def _load_step_verify_meta(run_root: Path) -> dict[tuple[int, int], dict[str, Any]]:
    """Load verification metadata from ``steps/*.json`` (source of truth for debugging)."""
    steps_dir = run_root / "steps"
    if not steps_dir.is_dir():
        return {}
    loaded: dict[tuple[int, int], dict[str, Any]] = {}
    for path in sorted(steps_dir.glob("*.json")):
        stem = path.stem
        if "_" not in stem:
            continue
        left, right = stem.split("_", 1)
        try:
            key = (int(left), int(right))
        except ValueError:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            loaded[key] = _step_verify_meta_from_payload(payload)
    return loaded


def _load_instruction_groups(run_root: Path) -> list[dict[str, Any]]:
    hand_rows = _iter_hand_csv_rows(run_root)
    if not hand_rows:
        return []

    report = _load_session_report_data(run_root)
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    tool_results = (
        report.get("tool_results") if isinstance(report.get("tool_results"), list) else []
    )
    step_file_meta = _load_step_verify_meta(run_root)

    order: list[tuple[int, int]] = []
    goals: dict[tuple[int, int], str] = {}
    verify_meta: dict[tuple[int, int], dict[str, Any]] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        transcript_counter = step.get("transcript_counter")
        script_step_index = step.get("script_step_index")
        if not isinstance(transcript_counter, int) or not isinstance(script_step_index, int):
            continue
        key = (transcript_counter, script_step_index)
        if key not in goals:
            order.append(key)
        goal = step.get("goal")
        if isinstance(goal, str) and goal.strip():
            goals[key] = goal.strip()
        report_meta = _step_verify_meta_from_payload(step)
        file_meta = step_file_meta.get(key, {})
        # Prefer step JSON (always current) over an older report.json.
        verify_meta[key] = {**report_meta, **file_meta} if file_meta else report_meta

    for key, file_meta in step_file_meta.items():
        if key not in verify_meta:
            verify_meta[key] = file_meta

    operations = [
        _hand_operation_from_row(run_root=run_root, operation_number=index, row=row)
        for index, row in enumerate(hand_rows, start=1)
    ]

    if not order:
        return [{"goal": _FALLBACK_GOAL, "operations": operations}]

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {key: [] for key in order}
    ungrouped: list[dict[str, Any]] = []

    tool_key_by_timestamp: dict[str, tuple[int, int]] = {}
    for tool in tool_results:
        if not isinstance(tool, dict):
            continue
        transcript_counter = tool.get("transcript_counter")
        script_step_index = tool.get("script_step_index")
        timestamp_key = _normalize_timestamp_key(tool.get("timestamp_utc"))
        if (
            isinstance(transcript_counter, int)
            and isinstance(script_step_index, int)
            and timestamp_key is not None
        ):
            tool_key_by_timestamp[timestamp_key] = (transcript_counter, script_step_index)

    for operation in operations:
        timestamp_key = _normalize_timestamp_key(operation.get("timestamp"))
        key = tool_key_by_timestamp.get(timestamp_key or "")
        if key is not None and key in groups:
            groups[key].append(operation)
        else:
            ungrouped.append(operation)

    grouped: list[dict[str, Any]] = []
    for key in order:
        goal = goals.get(key) or f"指令 {key[0] + 1}"
        entry: dict[str, Any] = {"goal": goal, "operations": groups[key]}
        entry.update(verify_meta.get(key, {}))
        grouped.append(entry)

    if ungrouped:
        grouped.append({"goal": _UNGROUPED_GOAL, "operations": ungrouped})

    return grouped


def _render_hand_operation_html(*, run_root: Path, operation: dict[str, Any]) -> str:
    action = operation["action"]
    args = operation["args"]
    ok = operation["ok"]
    message = operation["message"]
    timestamp = operation["timestamp"]
    before = operation["before"]
    after = operation["after"]
    operation_number = operation["operation_number"]

    status_class = "ok" if ok else "fail"
    status_label = escape(_status_text(ok))
    action_label = escape(f"{_HAND_OP_PREFIX}{operation_number}：{action}")
    time_text = escape(_timestamp_text(timestamp)) or "—"
    message_text = escape(message or "（無）")
    instruction_meta = _render_instruction_meta_html(_instruction_meta_pairs(args))

    shots = _render_shot_html("動作前截圖", before, run_root) + _render_shot_html(
        "動作後截圖", after, run_root
    )

    return (
        f'<li class="hand-op">'
        f'<div class="hand-op-title">'
        f'<span class="hand-op-action">{action_label}</span>'
        f'<span class="badge {status_class}">{status_label}</span>'
        f"</div>"
        f'<div class="meta"><dl>'
        f"{instruction_meta}"
        f"<dt>狀態</dt><dd><span class=\"badge {status_class}\">{status_label}</span></dd>"
        f"<dt>時間</dt><dd>{time_text}</dd>"
        f"<dt>訊息</dt><dd>{message_text}</dd>"
        f"</dl></div>"
        f"{_render_args_html(_args_without_instructions(args))}"
        f'<div class="shots">{shots}</div>'
        f"</li>"
    )


def _verify_badge_class(
    *,
    verify: dict[str, Any] | None,
    status: str | None,
    has_failure: bool,
) -> str:
    if isinstance(verify, dict):
        if verify.get("accomplished") is True or verify.get("branch") == "advance":
            return "ok"
        if verify.get("accomplished") is False or verify.get("branch") in {
            "retry",
            "goto",
            "skip",
            "stop",
            "replan",
            "backtrack",
        }:
            return "fail"
    if isinstance(status, str):
        normalized = status.strip().lower()
        if normalized in {"completed", "success", "ok"}:
            return "ok"
        if normalized in {"failed", "verify_failed", "error"}:
            return "fail"
    return "fail" if has_failure else "neutral"


def _render_verify_panel_html(
    *,
    verify: Any = None,
    status: Any = None,
) -> str:
    """Render verifier decision for a scripted instruction group."""
    has_verify = isinstance(verify, dict)
    has_status = isinstance(status, str) and bool(status.strip())
    if not has_verify and not has_status:
        return ""

    rows: list[str] = []
    if has_status:
        rows.append(f"<dt>Status</dt><dd>{escape(status.strip())}</dd>")

    if has_verify:
        accomplished = verify.get("accomplished")
        if accomplished is True:
            accomplished_text = "true"
        elif accomplished is False:
            accomplished_text = "false"
        else:
            accomplished_text = "—"
        rows.append(f"<dt>Accomplished</dt><dd>{escape(accomplished_text)}</dd>")
        branch = verify.get("branch")
        rows.append(f"<dt>Branch</dt><dd>{escape(str(branch) if branch is not None else '—')}</dd>")
        target_step = verify.get("target_step")
        if target_step is not None:
            rows.append(f"<dt>Target step</dt><dd>{escape(str(target_step))}</dd>")
        if "clearly_unmet" in verify:
            clearly = verify.get("clearly_unmet")
            if clearly is True:
                clearly_text = "true"
            elif clearly is False:
                clearly_text = "false"
            else:
                clearly_text = "—"
            rows.append(f"<dt>Clearly unmet</dt><dd>{escape(clearly_text)}</dd>")
        reason = verify.get("reason") or verify.get("outcome")
        rows.append(f"<dt>Reason</dt><dd>{escape(str(reason) if reason else '—')}</dd>")
        updated_state = verify.get("updated_state")
        if isinstance(updated_state, str) and updated_state.strip():
            rows.append(f"<dt>Updated state</dt><dd>{escape(updated_state.strip())}</dd>")

    return (
        f'<div class="session-verify">'
        f'<div class="session-verify-title">驗證結果</div>'
        f'<div class="meta"><dl>{"".join(rows)}</dl></div>'
        f"</div>"
    )


def _render_instruction_group_html(
    *,
    run_root: Path,
    goal: str,
    operations: list[dict[str, Any]],
    step_number: int,
    expected_outcome: Any = None,
    verify: Any = None,
    status: Any = None,
) -> str:
    operation_count = len(operations)
    count_label = escape(f"{operation_count} 個動作")
    has_failure = any(not operation.get("ok", False) for operation in operations)
    verify_dict = verify if isinstance(verify, dict) else None
    status_text = status.strip() if isinstance(status, str) else None
    summary_badge_class = _verify_badge_class(
        verify=verify_dict,
        status=status_text,
        has_failure=has_failure,
    )
    step_label = escape(f"{step_number}.")
    has_expected = isinstance(expected_outcome, str) and bool(expected_outcome.strip())
    show_empty_expected = (not has_expected) and isinstance(verify_dict, dict)

    if isinstance(verify_dict, dict) and verify_dict.get("branch"):
        primary_badge = escape(str(verify_dict.get("branch")))
    elif status_text:
        primary_badge = escape(status_text)
    else:
        primary_badge = count_label

    badges = (
        f'<span class="instruction-badges">'
        f'<span class="badge {summary_badge_class}">{primary_badge}</span>'
    )
    if primary_badge != count_label:
        badges += f'<span class="badge neutral">{count_label}</span>'
    badges += "</span>"

    if has_expected:
        expected_summary = (
            f'<span class="instruction-expected">'
            f"預期結果：{escape(expected_outcome.strip())}"
            f"</span>"
        )
    elif show_empty_expected:
        expected_summary = (
            '<span class="instruction-expected instruction-expected-empty">'
            "預期結果：（無）"
            "</span>"
        )
    else:
        expected_summary = ""

    if operations:
        items = "".join(
            _render_hand_operation_html(run_root=run_root, operation=operation)
            for operation in operations
        )
        body = f'<ul class="hand-ops">{items}</ul>'
    else:
        body = '<p class="args-empty" style="padding: 1rem 1.5rem;">（無手部動作）</p>'

    verify_panel = _render_verify_panel_html(
        verify=verify,
        status=status,
    )

    return (
        f'<details class="instruction-group">'
        f"<summary>"
        f'<span class="instruction-number">{step_label}</span>'
        f'<span class="instruction-summary-text">'
        f'<span class="instruction-title">{escape(goal)}</span>'
        f"{expected_summary}"
        f"</span>"
        f"{badges}"
        f"</summary>"
        f"{verify_panel}"
        f"{body}"
        f"</details>"
    )


def _format_duration_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    seconds = max(0.0, float(value))
    if seconds >= 60:
        minutes = int(seconds // 60)
        rem = seconds - minutes * 60
        return f"{minutes}m {rem:.0f}s"
    return f"{seconds:.1f}s"


def _load_run_report(run_root: Path) -> dict[str, Any] | None:
    path = run_root / "report.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _reason_badge_class(reason: str, *, has_failures: bool) -> str:
    if has_failures:
        return "fail"
    normalized = reason.strip().lower()
    if normalized in {"completed", "success", "ok"}:
        return "ok"
    if normalized:
        return "neutral"
    return "neutral"


def _resolve_index_script_name(run_root: Path, report: dict[str, Any] | None) -> str:
    """Prefer the saved instruction/script filename, then smart goal text, then folder name."""

    def _filename_from(payload: dict[str, Any] | None) -> str | None:
        if not isinstance(payload, dict):
            return None
        path = payload.get("script_path")
        if isinstance(path, str) and path.strip():
            return script_display_name(Path(path.strip()))
        name = payload.get("script_name")
        if isinstance(name, str) and name.strip() and name.strip() != "智能模式":
            return name.strip()
        return None

    def _smart_goal_from(payload: dict[str, Any] | None) -> str | None:
        if not isinstance(payload, dict):
            return None
        goal = payload.get("smart_goal")
        if isinstance(goal, str) and goal.strip():
            return goal.strip()
        return None

    filename = _filename_from(report)
    if filename:
        return filename
    goal = _smart_goal_from(report)
    if goal:
        return goal
    if isinstance(report, dict):
        name = report.get("script_name")
        if isinstance(name, str) and name.strip():
            return name.strip()

    meta = _resolve_script_metadata(run_root)
    filename = _filename_from(meta)
    if filename:
        return filename
    goal = _smart_goal_from(meta)
    if goal:
        return goal
    if meta.get("script_name"):
        return meta["script_name"]

    log_path = run_root / "run.log"
    if log_path.is_file():
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if _QUEUE_SCRIPT_LOG_MARKER in line:
                    name = line.split(_QUEUE_SCRIPT_LOG_MARKER, 1)[1].strip()
                    if name:
                        return name
        except OSError:
            pass

    return run_root.name


def _resolve_index_run_datetime(run_root: Path, report: dict[str, Any] | None) -> str:
    if isinstance(report, dict):
        for key in ("started_at_utc", "generated_at_utc"):
            value = report.get(key)
            if isinstance(value, str) and value.strip():
                formatted = _timestamp_text(value)
                if formatted:
                    return formatted
        steps = report.get("steps")
        if isinstance(steps, list):
            started = _resolve_started_at_utc(run_root, steps)
            if started:
                formatted = _timestamp_text(started)
                if formatted:
                    return formatted

    started = _resolve_started_at_utc(run_root, [])
    if started:
        formatted = _timestamp_text(started)
        if formatted:
            return formatted

    match = _RUN_FOLDER_TS_RE.match(run_root.name)
    if match is not None:
        date_part, time_part = match.groups()
        try:
            dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            )
            formatted = _timestamp_text(dt)
            if formatted:
                return formatted
        except ValueError:
            pass

    return "—"


def _sort_attr(value: Any) -> str:
    """Render a ``data-sort`` attribute for client-side table sorting."""
    if value is None:
        return ' data-sort=""'
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
    else:
        text = str(value)
    return f' data-sort="{escape(text, quote=True)}"'


def _render_index_row(run_root: Path) -> str:
    run_id = run_root.name
    href = escape(f"{quote(run_id, safe='')}/{_HTML_NAME}", quote=True)
    report = _load_run_report(run_root)
    script_name_raw = _resolve_index_script_name(run_root, report)
    script_name = escape(script_name_raw)
    run_time_raw = _resolve_index_run_datetime(run_root, report)
    run_time = escape(run_time_raw)
    run_id_title = escape(run_id, quote=True)
    summary = report.get("summary") if isinstance(report, dict) else None
    if not isinstance(summary, dict):
        summary = {}

    reason = ""
    if isinstance(report, dict):
        raw_reason = report.get("session_end_reason")
        if isinstance(raw_reason, str):
            reason = raw_reason.strip()

    step_count = summary.get("step_count")
    tool_count = summary.get("tool_call_count")
    failed_steps = summary.get("failed_step_count")
    failed_tools = summary.get("failed_tool_count")
    duration_raw = summary.get("total_duration_seconds")
    duration = _format_duration_seconds(duration_raw)

    failed_step_n = failed_steps if isinstance(failed_steps, int) else 0
    failed_tool_n = failed_tools if isinstance(failed_tools, int) else 0
    has_failures = failed_step_n > 0 or failed_tool_n > 0

    reason_label = escape(reason) if reason else "—"
    reason_class = _reason_badge_class(reason, has_failures=has_failures)
    reason_html = (
        f'<span class="badge {reason_class}">{reason_label}</span>'
        if reason
        else "—"
    )

    def _count_text(value: Any) -> str:
        return escape(str(value)) if isinstance(value, int) else "—"

    duration_sort = (
        float(duration_raw) if isinstance(duration_raw, (int, float)) else None
    )

    script_label = escape(script_name_raw, quote=True)
    return (
        "<tr>"
        f'<td class="select-col" data-label="選取" data-sort="">'
        f'<input type="checkbox" class="select-run" data-run-id="{run_id_title}" '
        f'data-run-label="{script_label}" '
        f'aria-label="選取報告 {run_id_title}"></td>'
        f'<td data-label="執行"{_sort_attr(script_name_raw)}>'
        f'<a href="{href}" title="{run_id_title}">{script_name}</a></td>'
        f'<td data-label="時間"{_sort_attr(run_time_raw)}>{run_time}</td>'
        f'<td data-label="結束原因"{_sort_attr(reason or "")}>{reason_html}</td>'
        f'<td data-label="步驟"{_sort_attr(step_count if isinstance(step_count, int) else None)}>'
        f"{_count_text(step_count)}</td>"
        f'<td data-label="工具"{_sort_attr(tool_count if isinstance(tool_count, int) else None)}>'
        f"{_count_text(tool_count)}</td>"
        f'<td data-label="失敗步驟"'
        f'{_sort_attr(failed_steps if isinstance(failed_steps, int) else None)}>'
        f"{_count_text(failed_steps)}</td>"
        f'<td data-label="失敗工具"'
        f'{_sort_attr(failed_tools if isinstance(failed_tools, int) else None)}>'
        f"{_count_text(failed_tools)}</td>"
        f'<td data-label="耗時"{_sort_attr(duration_sort)}>{escape(duration)}</td>'
        f'<td data-label="操作" class="actions" data-sort="">'
        f'<button type="button" class="bug-run" data-run-id="{run_id_title}" '
        f'data-run-label="{script_label}" '
        f'title="回報 bug" '
        f'aria-label="回報 bug {run_id_title}">🐛</button>'
        f'<button type="button" class="delete-run" data-run-id="{run_id_title}" '
        f'data-run-label="{script_label}" '
        f'title="刪除報告" aria-label="刪除報告 {run_id_title}">🗑</button></td>'
        "</tr>"
    )


def _iter_report_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    found: list[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir() or _is_recording_run_dir(child) or _is_smart_run_dir(child):
            continue
        if (child / _HTML_NAME).is_file():
            found.append(child)
    found.sort(key=lambda path: (path.name, path.stat().st_mtime), reverse=True)
    return found


def _iter_smart_report_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    found: list[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir() or _is_recording_run_dir(child) or not _is_smart_run_dir(child):
            continue
        if (child / _HTML_NAME).is_file():
            found.append(child)
    found.sort(key=lambda path: (path.name, path.stat().st_mtime), reverse=True)
    return found


def _is_recording_run_dir(run_root: Path) -> bool:
    """Recording folders are identified by ``session.json``, not a name prefix."""
    return (Path(run_root) / "session.json").is_file()


def _is_smart_run_dir(run_root: Path) -> bool:
    if run_root.name.startswith("smart_"):
        return True
    if (run_root / "smart_state.json").is_file():
        return True
    report = _load_run_report(run_root)
    if not isinstance(report, dict):
        return False
    if report.get("run_mode") == "smart":
        return True
    if isinstance(report.get("smart_cycles"), list):
        return True
    goal = report.get("smart_goal")
    return isinstance(goal, str) and bool(goal.strip())


def _iter_recording_source_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    found: list[Path] = []
    for child in runs_root.iterdir():
        if child.is_dir() and _is_recording_run_dir(child) and (child / "session.json").is_file():
            found.append(child)
    found.sort(key=lambda path: (path.name, path.stat().st_mtime), reverse=True)
    return found


def _iter_recording_report_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.is_dir():
        return []
    found: list[Path] = []
    for child in runs_root.iterdir():
        if child.is_dir() and _is_recording_run_dir(child) and (child / _RECORDING_HTML_NAME).is_file():
            found.append(child)
    found.sort(key=lambda path: (path.name, path.stat().st_mtime), reverse=True)
    return found


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_session_manifest(run_root: Path) -> dict[str, Any] | None:
    return _load_json_dict(run_root / "session.json")


def _resolve_recording_screenshot(raw: str | None, run_root: Path) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    for resolved in (
        run_root / candidate if not candidate.is_absolute() else None,
        run_root / "screenshots" / candidate.name,
    ):
        if resolved is not None and resolved.is_file():
            return resolved
    return None


def _recording_kind_label(kind: str, click_count: Any = None) -> str:
    if kind == "click":
        try:
            count = int(click_count) if click_count is not None else None
        except (TypeError, ValueError):
            count = None
        if count is not None and count >= 2:
            return f"連按{count}下"
    return _RECORDING_KIND_LABELS.get(kind, kind or "事件")


def _format_xy(value: Any) -> str | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return f"({int(value[0])}, {int(value[1])})"
        except (TypeError, ValueError):
            return f"({value[0]}, {value[1]})"
    return None


def _recording_duration_seconds(manifest: dict[str, Any] | None) -> float | None:
    if not isinstance(manifest, dict):
        return None
    started = manifest.get("started_at_utc")
    stopped = manifest.get("stopped_at_utc")
    if not isinstance(started, str) or not isinstance(stopped, str):
        return None
    try:
        start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        stop_dt = datetime.fromisoformat(stopped.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (stop_dt - start_dt).total_seconds())


def _resolve_recording_datetime(run_root: Path, manifest: dict[str, Any] | None) -> str:
    if isinstance(manifest, dict):
        for key in ("started_at_utc", "stopped_at_utc"):
            value = manifest.get(key)
            if isinstance(value, str) and value.strip():
                formatted = _timestamp_text(value)
                if formatted:
                    return formatted
    return _resolve_index_run_datetime(run_root, None)


def recording_event_json_paths(run_root: Path) -> list[Path]:
    """Event JSON paths in ``session.json`` order, or filename order as fallback."""
    run_root = Path(run_root)
    manifest = _load_session_manifest(run_root)
    if isinstance(manifest, dict) and isinstance(manifest.get("events"), list):
        event_paths: list[Path] = []
        seen: set[str] = set()
        for item in manifest["events"]:
            if not isinstance(item, str) or not item.strip():
                continue
            path = run_root / item
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            event_paths.append(path)
        return event_paths

    events_dir = run_root / "events"
    if events_dir.is_dir():
        return sorted(path for path in events_dir.glob("event_*.json") if path.is_file())
    return []


def _load_recording_events(run_root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in recording_event_json_paths(run_root):
        payload = _load_json_dict(path)
        if payload is not None:
            events.append(payload)
    return events


def _load_recording_analysis(run_root: Path, event_index: int) -> dict[str, Any] | None:
    return _load_json_dict(run_root / "analysis" / f"event_{event_index:03d}.json")


def _selected_landmark_labels(instruction: str, location: str) -> set[str]:
    from src.common.nearby_side import extract_nearby_hints_by_location

    buckets = extract_nearby_hints_by_location(instruction)
    return {hint.label for hint in buckets.get(location, [])}


_LANDMARK_SIDE_GROUP_ORDER: tuple[str | None, ...] = (
    "left",
    "right",
    "above",
    "below",
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "inside",
    None,
)

_LANDMARK_SIDE_GROUP_TITLES: dict[str | None, str] = {
    "left": "左邊",
    "right": "右邊",
    "above": "上面",
    "below": "下面",
    "upper_left": "左上方",
    "upper_right": "右上方",
    "lower_left": "左下方",
    "lower_right": "右下方",
    "inside": "裡面",
    None: "其他",
}


def _landmark_option_side_key(option: dict[str, Any]) -> str | None:
    side = option.get("side")
    if side is None or side == "":
        return None
    return str(side)


def _render_landmark_checkbox_items(
    *,
    group_key: str,
    options: list[dict[str, Any]],
    selected_labels: set[str],
) -> list[str]:
    items: list[str] = []
    for option in options:
        label = str(option.get("label") or "")
        if not label:
            continue
        display = str(option.get("display") or label)
        side = option.get("side")
        side_attr = "" if side is None else str(side)
        checked = " checked" if label in selected_labels else ""
        items.append(
            "<li>"
            f'<label><input type="checkbox" data-landmark-group="{escape(group_key, quote=True)}" '
            f'data-label="{escape(label, quote=True)}" '
            f'data-side="{escape(side_attr, quote=True)}"{checked}>'
            f"<span>{escape(display)}</span></label>"
            "</li>"
        )
    return items


def _render_landmark_group_html(
    *,
    title: str,
    group_key: str,
    options: list[dict[str, Any]],
    selected_labels: set[str],
) -> str:
    if not options:
        return ""

    buckets: dict[str | None, list[dict[str, Any]]] = {}
    for option in options:
        if not str(option.get("label") or ""):
            continue
        buckets.setdefault(_landmark_option_side_key(option), []).append(option)

    side_sections: list[str] = []
    for side_key in _LANDMARK_SIDE_GROUP_ORDER:
        side_options = buckets.get(side_key) or []
        if not side_options:
            continue
        items = _render_landmark_checkbox_items(
            group_key=group_key,
            options=side_options,
            selected_labels=selected_labels,
        )
        if not items:
            continue
        side_title = _LANDMARK_SIDE_GROUP_TITLES[side_key]
        side_sections.append(
            f'<div class="landmarks-side-group" data-side-group="{escape(side_key or "", quote=True)}">'
            f'<div class="landmarks-side-label">{escape(side_title)}</div>'
            f'<ul class="landmarks-list">{"".join(items)}</ul>'
            f"</div>"
        )
    if not side_sections:
        return ""
    return (
        f'<div class="landmarks-group">'
        f'<div class="landmarks-group-label">{escape(title)}</div>'
        f'<div class="landmarks-side-groups">{"".join(side_sections)}</div>'
        f"</div>"
    )


def _render_primary_target_group_html(
    *,
    title: str,
    group_key: str,
    event_index: int,
    options: list[dict[str, Any]],
) -> str:
    """Radio list for swapping the primary click/drag target (needs ≥2 options)."""
    if len(options) < 2:
        return ""
    name = f"primary-{group_key}-{event_index}"
    items: list[str] = []
    for option in options:
        try:
            index = int(option.get("index"))
        except (TypeError, ValueError):
            continue
        label = str(option.get("label") or "")
        if not label:
            continue
        display = str(option.get("display") or label)
        checked = " checked" if index == 0 else ""
        items.append(
            "<li>"
            f'<label><input type="radio" name="{escape(name, quote=True)}" '
            f'data-primary-group="{escape(group_key, quote=True)}" '
            f'data-primary-index="{index}" '
            f'data-label="{escape(label, quote=True)}"{checked}>'
            f"<span>{escape(display)}</span></label>"
            "</li>"
        )
    if len(items) < 2:
        return ""
    return (
        f'<div class="landmarks-group">'
        f'<div class="landmarks-group-label">{escape(title)}</div>'
        f'<ul class="landmarks-list">{"".join(items)}</ul>'
        f"</div>"
    )


def _render_landmarks_panel_html(
    *,
    run_root: Path,
    event_index: int,
    kind: str,
    instruction: str,
) -> str:
    if kind not in {"click", "double_click", "triple_click", "right_click", "middle_click", "scroll", "drag", "hold"}:
        return ""
    if not instruction.strip():
        return ""
    from src.recorder.vision_context import (
        load_recording_landmark_options,
        load_recording_primary_target_options,
    )

    primary_groups = load_recording_primary_target_options(
        run_root,
        event_index,
        kind=kind,
    )
    landmark_groups = load_recording_landmark_options(
        run_root,
        event_index,
        kind=kind,
        instruction=instruction,
    )
    start_primary = primary_groups.get("start") or []
    end_primary = primary_groups.get("end") or []
    start_options = landmark_groups.get("start") or []
    end_options = landmark_groups.get("end") or []

    groups_html = ""
    if kind == "drag":
        groups_html += _render_primary_target_group_html(
            title="起點目標",
            group_key="start",
            event_index=event_index,
            options=start_primary,
        )
        groups_html += _render_landmark_group_html(
            title="起點地標",
            group_key="start",
            options=start_options,
            selected_labels=_selected_landmark_labels(instruction, "起點"),
        )
        groups_html += _render_primary_target_group_html(
            title="終點目標",
            group_key="end",
            event_index=event_index,
            options=end_primary,
        )
        groups_html += _render_landmark_group_html(
            title="終點地標",
            group_key="end",
            options=end_options,
            selected_labels=_selected_landmark_labels(instruction, "終點"),
        )
    else:
        groups_html += _render_primary_target_group_html(
            title="點擊目標",
            group_key="start",
            event_index=event_index,
            options=start_primary,
        )
        groups_html += _render_landmark_group_html(
            title="附近地標",
            group_key="start",
            options=start_options,
            selected_labels=_selected_landmark_labels(instruction, "附近"),
        )

    if not groups_html:
        return ""

    has_primary = (
        len(start_primary) >= 2
        or (kind == "drag" and len(end_primary) >= 2)
    )
    panel_title = "目標與地標" if has_primary else "附近地標"
    button_title = (
        "依選取目標與地標重新產生指令"
        if has_primary
        else "依勾選地標重新產生指令"
    )
    return (
        f'<div class="landmarks">'
        f'<div class="landmarks-title">{escape(panel_title)}</div>'
        f'<div class="landmarks-groups">{groups_html}</div>'
        f'<button type="button" class="apply-landmarks" title="{escape(button_title, quote=True)}">'
        f"套用</button>"
        f'<span class="landmarks-status" aria-live="polite"></span>'
        f"</div>"
    )


def _yolo_ocr_payload_has_candidates(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("yolo_error"):
        return False
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        return True
    count = payload.get("detection_count")
    return isinstance(count, int) and count > 0


def _recording_yolo_ocr_failed(run_root: Path, event_index: int, kind: str) -> bool:
    from src.recorder.vision_context import load_yolo_ocr_payload

    start = load_yolo_ocr_payload(run_root, event_index, suffix="")
    if _yolo_ocr_payload_has_candidates(start):
        return False
    if kind == "drag":
        end = load_yolo_ocr_payload(run_root, event_index, suffix="_end")
        return not _yolo_ocr_payload_has_candidates(end)
    return True


def _render_yolo_retry_panel_html(
    *,
    run_root: Path,
    event_index: int,
    kind: str,
) -> str:
    from src.recorder.models import POINTER_EVENT_KINDS
    from src.recorder.vision_context import load_yolo_ocr_payload

    if kind not in POINTER_EVENT_KINDS:
        return ""
    failed = _recording_yolo_ocr_failed(run_root, event_index, kind)
    payload = load_yolo_ocr_payload(run_root, event_index, suffix="")
    error_text = ""
    if isinstance(payload, dict):
        raw_error = payload.get("yolo_error")
        if isinstance(raw_error, str) and raw_error.strip():
            error_text = raw_error.strip()
    if failed:
        note = "分析時 YOLO/OCR 沒有偵測到目標（常為 Triton 逾時）。可重新偵測後重建指令。"
        if error_text:
            note = f"分析時 YOLO/OCR 失敗：{error_text}"
        title = "YOLO/OCR 未偵測到目標"
        failed_class = " failed"
    else:
        note = "用此步驟的動作前截圖再跑一次 YOLO/OCR，並依新結果重建指令。"
        title = "YOLO/OCR"
        failed_class = ""
    return (
        f'<div class="vision-retry{failed_class}">'
        f'<div class="vision-retry-title">{escape(title)}</div>'
        f'<p class="vision-retry-note">{escape(note)}</p>'
        f'<button type="button" class="rerun-yolo-ocr" title="重新偵測 YOLO/OCR">'
        f"重新偵測 YOLO/OCR</button>"
        f'<span class="vision-retry-status" aria-live="polite"></span>'
        f"</div>"
    )


def _typed_text_candidates(
    event: dict[str, Any],
    analysis: dict[str, Any] | None,
    instruction: str,
) -> tuple[str, list[str], str, str]:
    """Return ``(recorded, ocr_options, active, active_source)`` for the typed-text editor."""
    from src.recorder.analyze import typed_text_from_instruction
    from src.recorder.text_resolve import _strip_ocr_caret

    recorded = ""
    ocr_options: list[str] = []
    active = ""
    active_source = "recorded"

    resolution: dict[str, Any] | None = None
    if isinstance(analysis, dict):
        raw_resolution = analysis.get("text_resolution")
        if isinstance(raw_resolution, dict):
            resolution = raw_resolution

    if resolution is not None:
        recorded = str(resolution.get("recorded_text") or "").strip()
        raw_options = resolution.get("ocr_options")
        if isinstance(raw_options, list):
            for item in raw_options:
                text = _strip_ocr_caret(str(item or "").strip())
                if text and text not in ocr_options:
                    ocr_options.append(text)
        ocr = str(resolution.get("ocr_text") or "").strip()
        if not ocr and resolution.get("source") == "ocr":
            ocr = str(resolution.get("resolved_text") or "").strip()
        ocr = _strip_ocr_caret(ocr)
        if ocr and ocr not in ocr_options:
            ocr_options.insert(0, ocr)
        resolved = str(resolution.get("resolved_text") or "").strip()
        source = str(resolution.get("source") or "")
        if source != "user":
            resolved = _strip_ocr_caret(resolved)
        if source == "user" and resolved:
            active = resolved
            if resolved in ocr_options:
                active_source = "ocr"
            elif resolved == recorded:
                active_source = "recorded"
            else:
                active_source = "custom"
        elif recorded:
            active = recorded
            active_source = "recorded"
        elif ocr_options:
            active = ocr_options[0]
            active_source = "ocr"

    if not recorded:
        text = event.get("text")
        if isinstance(text, str):
            recorded = text.strip()

    if not active:
        extracted = typed_text_from_instruction(instruction) if instruction else None
        if extracted:
            active = extracted
            if extracted == recorded:
                active_source = "recorded"
            elif extracted in ocr_options:
                active_source = "ocr"
            else:
                active_source = "custom"
        elif recorded:
            active = recorded
            active_source = "recorded"
        elif ocr_options:
            active = ocr_options[0]
            active_source = "ocr"

    return recorded, ocr_options, active, active_source


def _typed_text_for_editor(
    event: dict[str, Any],
    analysis: dict[str, Any] | None,
    instruction: str,
) -> str:
    _recorded, _ocr_options, active, _active_source = _typed_text_candidates(
        event,
        analysis,
        instruction,
    )
    return active


def _render_typed_text_choice_button(
    *,
    label: str,
    text: str,
    source: str,
    selected: bool,
) -> str:
    if not text:
        return ""
    selected_class = " selected" if selected else ""
    return (
        f'<button type="button" class="typed-text-choice{selected_class}" '
        f'data-source="{escape(source)}" data-text="{escape(text, quote=True)}" '
        f'title="使用{escape(label)}">'
        f"{escape(label)}：{escape(text)}</button>"
    )


def _render_typed_text_panel_html(
    *,
    event: dict[str, Any],
    analysis: dict[str, Any] | None,
    instruction: str,
) -> str:
    kind = str(event.get("kind") or "")
    if kind != "text_input":
        return ""
    recorded, ocr_options, value, active_source = _typed_text_candidates(
        event,
        analysis,
        instruction,
    )
    choice_parts: list[str] = []
    for ocr_text in ocr_options:
        choice_parts.append(
            _render_typed_text_choice_button(
                label="OCR",
                text=ocr_text,
                source="ocr",
                selected=active_source == "ocr" and value == ocr_text,
            )
        )
    choice_parts.append(
        _render_typed_text_choice_button(
            label="鍵盤",
            text=recorded,
            source="recorded",
            selected=active_source == "recorded",
        )
    )
    choices = "".join(choice_parts)
    choices_html = (
        f'<div class="typed-text-choices">{choices}</div>' if choices else ""
    )
    return (
        f'<div class="typed-text">'
        f'<div class="typed-text-title">輸入文字</div>'
        f"{choices_html}"
        f'<div class="typed-text-row">'
        f'<input type="text" class="typed-text-input" value="{escape(value, quote=True)}" '
        f'spellcheck="false" autocomplete="off" aria-label="輸入文字">'
        f'<button type="button" class="apply-typed-text" title="儲存修改後的輸入文字">'
        f"套用</button>"
        f'<span class="typed-text-status" aria-live="polite"></span>'
        f"</div>"
        f"</div>"
    )


def _render_expected_outcome_panel_html(*, expected_outcome: str, show: bool) -> str:
    if not show:
        return ""
    return (
        f'<div class="expected-outcome">'
        f'<div class="expected-outcome-title">預期結果</div>'
        f'<div class="expected-outcome-row">'
        f'<textarea class="expected-outcome-input" rows="2" '
        f'spellcheck="false" aria-label="預期結果">'
        f"{escape(expected_outcome)}</textarea>"
        f'<button type="button" class="apply-expected-outcome" '
        f'title="儲存修改後的預期結果">套用</button>'
        f'<span class="expected-outcome-status" aria-live="polite"></span>'
        f"</div>"
        f"</div>"
    )


def _render_step_instruction_panel_html(*, instruction: str) -> str:
    return (
        f'<div class="step-instruction">'
        f'<div class="step-instruction-title">指令</div>'
        f'<div class="step-instruction-row">'
        f'<textarea class="step-instruction-input" rows="2" '
        f'spellcheck="false" aria-label="指令">'
        f"{escape(instruction)}</textarea>"
        f'<button type="button" class="apply-step-instruction" '
        f'title="儲存修改後的指令">套用</button>'
        f'<span class="step-instruction-status" aria-live="polite"></span>'
        f"</div>"
        f"</div>"
    )


def _recording_add_dialog_html() -> str:
    return """
<div class="add-step-dialog-backdrop" id="add-step-dialog" hidden>
  <div class="add-step-dialog" role="dialog" aria-modal="true" aria-labelledby="add-step-title">
    <h2 id="add-step-title">新增步驟</h2>
    <form>
      <div class="add-step-field">
        <label for="add-step-kind">種類</label>
        <select id="add-step-kind" name="kind">
          <option value="click">點擊</option>
          <option value="double_click">雙擊</option>
          <option value="right_click">右鍵點擊</option>
          <option value="text_input">輸入文字</option>
          <option value="key_press">按鍵</option>
          <option value="hotkey">快捷鍵</option>
          <option value="scroll">捲動</option>
          <option value="wait">等待</option>
          <option value="manual">自訂指令</option>
        </select>
      </div>
      <div class="add-step-field" data-kind-field="text_input" hidden>
        <label for="add-step-text">輸入文字</label>
        <input id="add-step-text" name="text" type="text" spellcheck="false" autocomplete="off">
      </div>
      <div class="add-step-field" data-kind-field="key_press" hidden>
        <label for="add-step-key">按鍵</label>
        <input id="add-step-key" name="key" type="text" spellcheck="false" autocomplete="off" placeholder="Enter">
      </div>
      <div class="add-step-field" data-kind-field="hotkey" hidden>
        <label for="add-step-keys">快捷鍵</label>
        <input id="add-step-keys" name="keys" type="text" spellcheck="false" autocomplete="off" placeholder="Ctrl+S">
      </div>
      <div class="add-step-field" data-kind-field="scroll" hidden>
        <label for="add-step-scroll">方向</label>
        <select id="add-step-scroll" name="scroll_delta">
          <option value="-1">向下</option>
          <option value="1">向上</option>
        </select>
      </div>
      <div class="add-step-field" data-kind-field="wait" hidden>
        <label for="add-step-wait">秒數</label>
        <input id="add-step-wait" name="duration_seconds" type="number" min="1" step="1" value="1">
      </div>
      <div class="add-step-field">
        <label for="add-step-instruction">指令</label>
        <textarea id="add-step-instruction" name="instruction" rows="3" spellcheck="false"></textarea>
      </div>
      <div class="add-step-field">
        <label for="add-step-outcome">預期結果（選填）</label>
        <textarea id="add-step-outcome" name="expected_outcome" rows="2" spellcheck="false"></textarea>
      </div>
      <p class="add-step-status" aria-live="polite"></p>
      <div class="add-step-actions">
        <button type="button" class="add-step-cancel">取消</button>
        <button type="submit" class="add-step-submit">新增</button>
      </div>
    </form>
  </div>
</div>
""".strip()


def _render_recording_event_html(
    *,
    run_root: Path,
    event: dict[str, Any],
    next_event: dict[str, Any] | None = None,
    display_index: int | None = None,
) -> str:
    raw_index = event.get("index")
    index = raw_index if isinstance(raw_index, int) else 0
    kind = str(event.get("kind") or "")
    kind_label = _recording_kind_label(kind, event.get("click_count"))
    analysis = _load_recording_analysis(run_root, index) if index else None
    instruction = ""
    if isinstance(analysis, dict):
        raw_instruction = analysis.get("instruction")
        if isinstance(raw_instruction, str) and raw_instruction.strip():
            instruction = raw_instruction.strip()

    title = instruction or kind_label
    shown_index = display_index if isinstance(display_index, int) and display_index > 0 else index
    step_label = escape(f"{shown_index}.")
    kind_badge = escape(kind_label)
    time_text = escape(_timestamp_text(event.get("timestamp_utc"))) or "—"
    run_id = escape(run_root.name, quote=True)

    expected_outcome = ""
    if isinstance(analysis, dict):
        raw_outcome = analysis.get("expected_outcome")
        if isinstance(raw_outcome, str) and raw_outcome.strip():
            expected_outcome = raw_outcome.strip()

    meta_rows: list[tuple[str, str]] = [("時間", time_text)]
    cursor = _format_xy(event.get("cursor_xy"))
    if cursor:
        meta_rows.append(("游標", escape(cursor)))
    end_xy = _format_xy(event.get("end_xy"))
    if end_xy:
        meta_rows.append(("終點", escape(end_xy)))
    text = event.get("text")
    if kind != "text_input" and isinstance(text, str) and text:
        meta_rows.append(("文字", escape(text)))
    key = event.get("key")
    if isinstance(key, str) and key:
        meta_rows.append(("按鍵", escape(key)))
    keys = event.get("keys")
    if isinstance(keys, list) and keys:
        meta_rows.append(("快捷鍵", escape("+".join(str(item) for item in keys))))
    scroll_delta = event.get("scroll_delta")
    if isinstance(scroll_delta, int):
        meta_rows.append(("捲動", escape(str(scroll_delta))))
    duration_seconds = event.get("duration_seconds")
    if isinstance(duration_seconds, (int, float)):
        meta_rows.append(("等待", escape(str(duration_seconds))))
    window_title = event.get("target_window_title")
    if isinstance(window_title, str) and window_title.strip():
        meta_rows.append(("視窗", escape(window_title.strip())))

    meta_html = "".join(f"<dt>{escape(label)}</dt><dd>{value}</dd>" for label, value in meta_rows)

    before = _resolve_recording_screenshot(str(event.get("screenshot_path") or ""), run_root)
    after: Path | None = None
    if next_event is not None:
        after = _resolve_recording_screenshot(
            str(next_event.get("screenshot_path") or ""),
            run_root,
        )
    if after is None:
        after = _resolve_recording_screenshot(
            str(event.get("end_screenshot_path") or ""),
            run_root,
        )
    shots = _render_shot_html("動作前截圖", before, run_root) + _render_shot_html(
        "動作後截圖", after, run_root
    )

    landmarks_html = _render_landmarks_panel_html(
        run_root=run_root,
        event_index=index,
        kind=kind,
        instruction=instruction,
    )
    yolo_retry_html = _render_yolo_retry_panel_html(
        run_root=run_root,
        event_index=index,
        kind=kind,
    )
    typed_text_html = _render_typed_text_panel_html(
        event=event,
        analysis=analysis if isinstance(analysis, dict) else None,
        instruction=instruction,
    )
    expected_outcome_html = _render_expected_outcome_panel_html(
        expected_outcome=expected_outcome,
        show=bool(instruction) or bool(expected_outcome),
    )
    instruction_html = _render_step_instruction_panel_html(instruction=instruction)

    copy_attr = escape(title, quote=True)
    outcome_attr = (
        f' data-expected-outcome="{escape(expected_outcome, quote=True)}"'
        if expected_outcome
        else ""
    )
    return (
        f'<details class="instruction-group" id="event-{index}" data-run-id="{run_id}" '
        f'data-event-index="{index}" data-kind="{escape(kind, quote=True)}">'
        f"<summary>"
        f'<span class="instruction-number">{step_label}</span>'
        f'<span class="instruction-title">{escape(title)}</span>'
        f'<span class="badge neutral">{kind_badge}</span>'
        f'<button type="button" class="copy-instruction" data-instruction="{copy_attr}"'
        f'{outcome_attr} '
        f'title="複製指令" aria-label="複製指令">複製</button>'
        f'<button type="button" class="add-instruction" '
        f'title="在此步驟後新增" aria-label="新增步驟">新增</button>'
        f'<button type="button" class="delete-instruction" '
        f'title="刪除指令" aria-label="刪除指令">刪除</button>'
        f"</summary>"
        f'<div class="meta" style="padding: 1rem 1.5rem 0;"><dl>{meta_html}</dl></div>'
        f'<div class="instruction-edit-panels">{instruction_html}{expected_outcome_html}</div>'
        f"{typed_text_html}"
        f"{yolo_retry_html}"
        f"{landmarks_html}"
        f'<div class="shots" style="padding: 0 1.5rem 1rem;">{shots}</div>'
        f'<div class="collapse-row">'
        f'<button type="button" class="collapse-instruction" '
        f'title="收合" aria-label="收合">▲</button>'
        f"</div>"
        f"</details>"
    )


def _render_recording_index_row(run_root: Path) -> str:
    run_id = run_root.name
    href = escape(f"{quote(run_id, safe='')}/{_RECORDING_HTML_NAME}", quote=True)
    manifest = _load_session_manifest(run_root)
    report = _load_run_report(run_root)
    run_time_raw = _resolve_recording_datetime(run_root, manifest)
    run_time = escape(run_time_raw)
    run_id_title = escape(run_id, quote=True)
    label = escape(run_id)
    label_attr = escape(run_id, quote=True)

    event_count: int | None = None
    if isinstance(manifest, dict) and isinstance(manifest.get("event_count"), int):
        event_count = manifest["event_count"]
    elif isinstance(report, dict) and isinstance(report.get("recorded"), int):
        event_count = report["recorded"]

    analyzed_count: int | None = None
    error_count: int | None = None
    if isinstance(report, dict):
        if isinstance(report.get("cached"), int):
            analyzed_count = report["cached"]
        elif isinstance(report.get("instructions"), list):
            analyzed_count = len(report["instructions"])
        errors = report.get("errors")
        if isinstance(errors, list):
            error_count = len(errors)

    duration_raw = _recording_duration_seconds(manifest)
    duration = _format_duration_seconds(duration_raw)

    def _count_text(value: Any) -> str:
        return escape(str(value)) if isinstance(value, int) else "—"

    return (
        "<tr>"
        f'<td class="select-col" data-label="選取" data-sort="">'
        f'<input type="checkbox" class="select-run" data-run-id="{run_id_title}" '
        f'data-run-label="{label_attr}" '
        f'aria-label="選取錄製 {run_id_title}"></td>'
        f'<td data-label="錄製"{_sort_attr(run_id)}>'
        f'<a href="{href}" title="{run_id_title}">{label}</a></td>'
        f'<td data-label="時間"{_sort_attr(run_time_raw)}>{run_time}</td>'
        f'<td data-label="事件"{_sort_attr(event_count)}>{_count_text(event_count)}</td>'
        f'<td data-label="已分析"{_sort_attr(analyzed_count)}>{_count_text(analyzed_count)}</td>'
        f'<td data-label="錯誤"{_sort_attr(error_count)}>{_count_text(error_count)}</td>'
        f'<td data-label="耗時"{_sort_attr(duration_raw)}>{escape(duration)}</td>'
        f'<td data-label="操作" class="actions" data-sort="">'
        f'<button type="button" class="bug-run" data-run-id="{run_id_title}" '
        f'data-run-label="{label_attr}" '
        f'title="回報 bug" '
        f'aria-label="回報 bug {run_id_title}">🐛</button>'
        f'<button type="button" class="delete-run" data-run-id="{run_id_title}" '
        f'data-run-label="{label_attr}" '
        f'title="刪除錄製" aria-label="刪除錄製 {run_id_title}">🗑</button></td>'
        "</tr>"
    )


def _render_bulk_bar() -> str:
    return (
        '<div class="bulk-bar" role="toolbar" aria-label="批次操作">'
        '<span class="bulk-count">已選 0 筆</span>'
        '<button type="button" class="bulk-bug" disabled>🐛 回報選取</button>'
        '<button type="button" class="bulk-delete" disabled>🗑 刪除選取</button>'
        "</div>"
    )


def _render_runs_tab_panel(
    run_dirs: list[Path],
    *,
    tab_id: str = "runs",
    hidden: bool = False,
) -> str:
    is_smart = tab_id == "smart"
    noun = "智能模式報告" if is_smart else "報告"
    name_column = "目標" if is_smart else "執行"
    select_aria = "全選智能模式報告" if is_smart else "全選報告"
    open_hint = "點選目標開啟步驟紀錄" if is_smart else "點選執行名稱開啟步驟紀錄"
    if run_dirs:
        rows = "".join(_render_index_row(run_dir) for run_dir in run_dirs)
        intro = (
            f"共 {len(run_dirs)} 筆{noun}。勾選多筆後可批次回報或刪除；"
            f"{open_hint}；"
            "點選欄位標題可排序；🐛 可回報 bug；"
            "垃圾桶可刪除整份報告資料夾。"
        )
        body = (
            f'<p class="intro">{intro}</p>\n'
            f"{_render_bulk_bar()}"
            '<table class="reports">'
            "<thead><tr>"
            '<th class="no-sort select-col" title="全選">'
            f'<input type="checkbox" class="select-all" aria-label="{select_aria}">'
            "</th>"
            f'<th data-type="text">{name_column}</th>'
            '<th data-type="text">時間</th>'
            '<th data-type="text">結束原因</th>'
            '<th data-type="num">步驟</th>'
            '<th data-type="num">工具</th>'
            '<th data-type="num">失敗步驟</th>'
            '<th data-type="num">失敗工具</th>'
            '<th data-type="num">耗時</th>'
            '<th class="no-sort">操作</th>'
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
        )
    else:
        body = (
            f'<p class="intro">共 0 筆{noun}。完成一次'
            f'{"智能模式" if is_smart else ""}執行後，報告會出現在此列表。</p>\n'
            f'<p class="empty">尚無{noun}。完成一次'
            f'{"智能模式" if is_smart else ""}執行後，報告會出現在此列表。</p>'
        )
    hidden_attr = " hidden" if hidden else ""
    return (
        f'<section class="tab-panel" data-tab="{tab_id}" id="tab-{tab_id}"{hidden_attr}>\n'
        f"{body}\n"
        f"</section>"
    )

def _render_recordings_tab_panel(recording_dirs: list[Path]) -> str:
    if recording_dirs:
        rows = "".join(_render_recording_index_row(run_dir) for run_dir in recording_dirs)
        intro = (
            f"共 {len(recording_dirs)} 筆錄製。勾選多筆後可批次回報或刪除；"
            "點選錄製名稱開啟事件紀錄；"
            "點選欄位標題可排序；🐛 可回報 bug；"
            "垃圾桶可刪除整份錄製資料夾。"
        )
        body = (
            f'<p class="intro">{intro}</p>\n'
            f"{_render_bulk_bar()}"
            '<table class="reports">'
            "<thead><tr>"
            '<th class="no-sort select-col" title="全選">'
            '<input type="checkbox" class="select-all" aria-label="全選錄製">'
            "</th>"
            '<th data-type="text">錄製</th>'
            '<th data-type="text">時間</th>'
            '<th data-type="num">事件</th>'
            '<th data-type="num">已分析</th>'
            '<th data-type="num">錯誤</th>'
            '<th data-type="num">耗時</th>'
            '<th class="no-sort">操作</th>'
            "</tr></thead>"
            f"<tbody>{rows}</tbody>"
            "</table>"
        )
    else:
        body = (
            '<p class="intro">共 0 筆錄製。完成一次錄製後，紀錄會出現在此列表。</p>\n'
            '<p class="empty">尚無錄製。完成一次錄製後，紀錄會出現在此列表。</p>'
        )
    return (
        f'<section class="tab-panel" data-tab="recordings" id="tab-recordings" hidden>\n'
        f"{body}\n"
        f"</section>"
    )


def _backfill_recording_html(runs_root: Path) -> None:
    for run_dir in _iter_recording_source_dirs(runs_root):
        write_recording_html_from_run(run_dir, update_index=False)


def write_runs_index_html(runs_root: Path) -> Path:
    """Build ``index.html`` with tabs for agent runs, smart mode, and recordings."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    _backfill_recording_html(runs_root)

    run_dirs = _iter_report_run_dirs(runs_root)
    smart_dirs = _iter_smart_report_run_dirs(runs_root)
    recording_dirs = _iter_recording_report_dirs(runs_root)
    tabs = (
        '<nav class="tabs" role="tablist" aria-label="報告類型">'
        '<button type="button" class="active" data-tab="runs" role="tab" aria-selected="true"'
        ' aria-controls="tab-runs">執行報告</button>'
        '<button type="button" data-tab="smart" role="tab" aria-selected="false"'
        ' aria-controls="tab-smart">智能模式</button>'
        '<button type="button" data-tab="recordings" role="tab" aria-selected="false"'
        ' aria-controls="tab-recordings">錄製紀錄</button>'
        "</nav>"
    )
    body = (
        f"{tabs}\n"
        f"{_render_runs_tab_panel(run_dirs)}\n"
        f"{_render_runs_tab_panel(smart_dirs, tab_id='smart', hidden=True)}\n"
        f"{_render_recordings_tab_panel(recording_dirs)}\n"
    )
    script = f"<script>\n{_INDEX_SCRIPT}\n</script>\n"
    title = "工作階段報告列表"
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{escape(title)}</title>\n"
        f"<style>\n{_INDEX_STYLE}\n</style>\n"
        "</head>\n<body>\n"
        f"<h1>{escape(title)}</h1>\n"
        f"{body}"
        f"{script}"
        "</body>\n</html>\n"
    )

    path = runs_index_html_path(runs_root)
    path.write_text(html, encoding="utf-8")
    return path


def _resolve_session_title(run_root: Path) -> str:
    """Build the page title from source script name and run datetime."""
    report = _load_run_report(run_root)
    script_name = _resolve_index_script_name(run_root, report)
    run_time = _resolve_index_run_datetime(run_root, report)
    if run_time and run_time != "—":
        return f"{script_name} · {run_time}"
    return script_name


def _resolve_recording_title(run_root: Path) -> str:
    manifest = _load_session_manifest(run_root)
    run_time = _resolve_recording_datetime(run_root, manifest)
    if run_time and run_time != "—":
        return f"{run_root.name} · {run_time}"
    return run_root.name


def _render_smart_cycles_html(
    report: dict[str, Any],
    *,
    run_root: Path,
    instruction_groups: list[dict[str, Any]],
) -> str:
    cycles = report.get("smart_cycles")
    if not isinstance(cycles, list) or not cycles:
        return ""
    goal = report.get("smart_goal")
    goal_html = (
        f'<p class="intro">智能模式目標：{escape(str(goal))}</p>\n'
        if isinstance(goal, str) and goal.strip()
        else ""
    )
    blocks: list[str] = [goal_html, '<section class="smart-cycles"><h2>Plan → Act → Verify</h2>']
    actor_group_index = 0
    for cycle in cycles:
        if not isinstance(cycle, dict):
            continue
        number = cycle.get("cycle", "?")
        plan = cycle.get("plan") if isinstance(cycle.get("plan"), dict) else {}
        act = cycle.get("act") if isinstance(cycle.get("act"), dict) else {}
        verify = cycle.get("verify") if isinstance(cycle.get("verify"), dict) else {}
        instruction = plan.get("instruction") or act.get("instruction") or "—"
        verify_branch = verify.get("branch") or "—"
        act_ok = act.get("ok")
        badge = "ok" if act_ok else ("fail" if act_ok is False else "neutral")
        tools_html = ""
        if act:
            group = (
                instruction_groups[actor_group_index]
                if actor_group_index < len(instruction_groups)
                else None
            )
            actor_group_index += 1
            if isinstance(group, dict):
                operations = (
                    group.get("operations") if isinstance(group.get("operations"), list) else []
                )
                operation_count = len(operations)
                if operations:
                    items = "".join(
                        _render_hand_operation_html(run_root=run_root, operation=operation)
                        for operation in operations
                    )
                    operations_html = f'<ul class="hand-ops">{items}</ul>'
                else:
                    operations_html = (
                        '<p class="args-empty" style="padding: 1rem 1.5rem;">'
                        "（無手部動作）</p>"
                    )
                tools_html = (
                    f'<div class="executed-tools">'
                    f"<h3>Executed tools ({operation_count})</h3>"
                    f"{operations_html}"
                    f"</div>"
                )
        blocks.append(
            f'<details class="instruction-group">'
            f"<summary>"
            f'<span class="instruction-number">{escape(str(number))}.</span>'
            f'<span class="instruction-title">{escape(str(instruction))}</span>'
            f'<span class="badge {badge}">{escape(str(verify_branch))}</span>'
            f"</summary>"
            f'<div class="meta smart-cycle-meta"><dl>'
            f"<dt>Plan</dt><dd>{escape(str(plan.get('rationale') or plan.get('status') or '—'))}</dd>"
            f"<dt>Expected</dt><dd>{escape(str(plan.get('expected_outcome') or '—'))}</dd>"
            f"<dt>Act</dt><dd>{escape(str(act.get('reason') or ('ok' if act_ok else 'fail' if act_ok is False else '—')))}</dd>"
            f"<dt>Verify</dt><dd>{escape(str(verify.get('reason') or verify.get('outcome') or '—'))}</dd>"
            f"<dt>Updated state</dt><dd>{escape(str(verify.get('updated_state') or '—'))}</dd>"
            f"</dl></div>"
            f"{tools_html}"
            f"</details>"
        )
    blocks.append("</section>")
    return "\n".join(blocks)


def write_session_html_from_run(run_root: Path) -> Path:
    """Build ``session_steps.html`` from ``hand.csv`` in a single pass (O(n)).

    Screenshots are referenced relatively (e.g. ``eye/<file>.png``) so the report stays tiny; keep
    the run folder together when sharing. Safe to call repeatedly and for rebuilding old runs, and
    handles both headered ``hand.csv`` files and legacy header-less ones.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    instruction_groups = _load_instruction_groups(run_root)
    report = _load_session_report_data(run_root)
    smart_cycles = report.get("smart_cycles") if isinstance(report, dict) else None
    smart_actor_count = (
        sum(
            1
            for cycle in smart_cycles
            if isinstance(cycle, dict) and isinstance(cycle.get("act"), dict) and cycle.get("act")
        )
        if isinstance(smart_cycles, list)
        else 0
    )
    remaining_groups = instruction_groups[min(smart_actor_count, len(instruction_groups)) :]
    groups_html = [
        _render_instruction_group_html(
            run_root=run_root,
            goal=group["goal"],
            operations=group["operations"],
            step_number=index,
            expected_outcome=group.get("expected_outcome"),
            verify=group.get("verify"),
            status=group.get("status"),
        )
        for index, group in enumerate(remaining_groups, start=smart_actor_count + 1)
    ]
    smart_html = _render_smart_cycles_html(
        report if isinstance(report, dict) else {},
        run_root=run_root,
        instruction_groups=instruction_groups,
    )

    title = escape(_resolve_session_title(run_root))
    body = smart_html + "\n" + "\n".join(groups_html)
    nav_href = "../index.html#smart" if _is_smart_run_dir(run_root) else "../index.html"
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_STYLE}\n</style>\n"
        "</head>\n<body>\n"
        f'<p class="nav"><a href="{nav_href}">← 報告列表</a></p>\n'
        f"<h1>{title}</h1>\n"
        '<p class="intro">依使用者指令分組的手部動作紀錄。點選指令可展開底下的動作列表。</p>\n'
        f"{body}\n"
        "</body>\n</html>\n"
    )

    path = session_html_path(run_root)
    path.write_text(html, encoding="utf-8")
    write_runs_index_html(run_root.parent)
    return path


def write_recording_html_from_run(run_root: Path, *, update_index: bool = True) -> Path:
    """Build ``recording_steps.html`` from recorded events and optional analysis."""
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    events = _load_recording_events(run_root)
    events_html = [
        _render_recording_event_html(
            run_root=run_root,
            event=event,
            next_event=events[index + 1] if index + 1 < len(events) else None,
            display_index=index + 1,
        )
        for index, event in enumerate(events)
    ]
    title = escape(_resolve_recording_title(run_root))
    body = "\n".join(events_html) if events_html else '<p class="empty">尚無錄製事件。</p>'
    copy_all_disabled = "" if events_html else " disabled"
    run_id_attr = escape(run_root.name, quote=True)
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_STYLE}\n.empty {{ color: #8c959f; font-style: italic; }}\n</style>\n"
        "</head>\n<body>\n"
        '<p class="nav"><a href="../index.html#recordings">← 報告列表</a></p>\n'
        f"<h1>{title}</h1>\n"
        '<p class="intro">依錄製事件排列的操作紀錄。點選事件可展開細節與截圖。</p>\n'
        f'<div class="recording-toolbar" data-run-id="{run_id_attr}">'
        f'<button type="button" class="copy-all-instructions"{copy_all_disabled} '
        'title="複製全部指令" aria-label="複製全部指令">複製全部指令</button>'
        '<button type="button" class="rename-recording" '
        'title="重新命名" aria-label="重新命名">重新命名</button>'
        '<button type="button" class="add-recording-step" '
        'title="新增步驟" aria-label="新增步驟">新增步驟</button>'
        "</div>\n"
        f"{body}\n"
        f"{_recording_add_dialog_html()}\n"
        f"<script>\n{_RECORDING_SCRIPT}\n</script>\n"
        "</body>\n</html>\n"
    )

    path = recording_html_path(run_root)
    path.write_text(html, encoding="utf-8")
    if update_index:
        write_runs_index_html(run_root.parent)
    return path
