from __future__ import annotations

import ast
import csv
import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

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
    "right_click": "右鍵點擊",
    "middle_click": "中鍵點擊",
    "drag": "拖曳",
    "scroll": "捲動",
    "text_input": "輸入文字",
    "key": "按鍵",
    "key_press": "按鍵",
    "hotkey": "快捷鍵",
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
.instruction-group[open] > summary { border-bottom: 1px solid #d0d7de; }
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
.executed-tools { border-top: 1px solid #d0d7de; }
.executed-tools h3 { font-size: 1rem; margin: 0; padding: 1rem 1.5rem 0; }
.meta { margin: 0 0 1rem; }
.meta dl { display: grid; grid-template-columns: max-content 1fr; gap: .25rem 1rem; margin: 0; }
.meta dt { color: #57606a; font-weight: 600; }
.meta dd { margin: 0; }
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

    function activate(tabId) {
      var resolved = tabId === "recordings" ? "recordings" : "runs";
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
    activate(hash === "recordings" ? "recordings" : "runs");
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

    headers.forEach(function (th, col) {
      if (th.classList.contains("no-sort")) return;
      th.classList.add("sortable");
      th.setAttribute("tabindex", "0");
      th.setAttribute("role", "columnheader");
      th.setAttribute("title", "點選排序");

      function sortByColumn() {
        var type = th.getAttribute("data-type") || "text";
        if (sortCol === col) {
          sortAsc = !sortAsc;
        } else {
          sortCol = col;
          sortAsc = true;
        }
        headers.forEach(function (header) {
          header.removeAttribute("aria-sort");
        });
        th.setAttribute("aria-sort", sortAsc ? "ascending" : "descending");
        var rows = Array.prototype.slice.call(tbody.rows);
        rows.sort(function (a, b) { return compareRows(a, b, type); });
        rows.forEach(function (row) { tbody.appendChild(row); });
      }

      th.addEventListener("click", sortByColumn);
      th.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          sortByColumn();
        }
      });
    });

    function introHelpText(count) {
      if (kind === "recordings") {
        return "共 " + count + " 筆錄製。勾選多筆後可批次回報或刪除；點選錄製名稱開啟事件紀錄；點選欄位標題可排序；🐛 可回報 bug；垃圾桶可刪除整份錄製資料夾。";
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


def _load_instruction_groups(run_root: Path) -> list[dict[str, Any]]:
    hand_rows = _iter_hand_csv_rows(run_root)
    if not hand_rows:
        return []

    report = _load_session_report_data(run_root)
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    tool_results = (
        report.get("tool_results") if isinstance(report.get("tool_results"), list) else []
    )

    order: list[tuple[int, int]] = []
    goals: dict[tuple[int, int], str] = {}
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
        grouped.append({"goal": goal, "operations": groups[key]})

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


def _render_instruction_group_html(
    *,
    run_root: Path,
    goal: str,
    operations: list[dict[str, Any]],
    step_number: int,
) -> str:
    operation_count = len(operations)
    count_label = escape(f"{operation_count} 個動作")
    has_failure = any(not operation.get("ok", False) for operation in operations)
    summary_badge_class = "fail" if has_failure else "neutral"
    step_label = escape(f"{step_number}.")

    if operations:
        items = "".join(
            _render_hand_operation_html(run_root=run_root, operation=operation)
            for operation in operations
        )
        body = f'<ul class="hand-ops">{items}</ul>'
    else:
        body = '<p class="args-empty" style="padding: 1rem 1.5rem;">（無手部動作）</p>'

    return (
        f'<details class="instruction-group">'
        f"<summary>"
        f'<span class="instruction-number">{step_label}</span>'
        f'<span class="instruction-title">{escape(goal)}</span>'
        f'<span class="badge {summary_badge_class}">{count_label}</span>'
        f"</summary>"
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
    if isinstance(report, dict):
        name = report.get("script_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
        path = report.get("script_path")
        if isinstance(path, str) and path.strip():
            return Path(path.strip()).name

    meta = _resolve_script_metadata(run_root)
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
    href = escape(f"{run_id}/{_HTML_NAME}", quote=True)
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
        if not child.is_dir() or _is_recording_run_dir(child):
            continue
        if (child / _HTML_NAME).is_file():
            found.append(child)
    found.sort(key=lambda path: (path.name, path.stat().st_mtime), reverse=True)
    return found


def _is_recording_run_dir(run_root: Path) -> bool:
    return run_root.name.startswith("recording_")


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
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for resolved in (
        run_root / candidate,
        run_root / "screenshots" / candidate.name,
    ):
        if resolved.is_file():
            return resolved
    return None


def _recording_kind_label(kind: str) -> str:
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


def _load_recording_events(run_root: Path) -> list[dict[str, Any]]:
    manifest = _load_session_manifest(run_root)
    event_paths: list[Path] = []
    if isinstance(manifest, dict):
        raw_events = manifest.get("events")
        if isinstance(raw_events, list):
            for item in raw_events:
                if isinstance(item, str) and item.strip():
                    event_paths.append(run_root / item)

    if not event_paths:
        events_dir = run_root / "events"
        if events_dir.is_dir():
            event_paths = sorted(events_dir.glob("event_*.json"))

    events: list[dict[str, Any]] = []
    for path in event_paths:
        payload = _load_json_dict(path)
        if payload is not None:
            events.append(payload)
    events.sort(key=lambda event: int(event.get("index", 0)) if isinstance(event.get("index"), int) else 0)
    return events


def _load_recording_analysis(run_root: Path, event_index: int) -> dict[str, Any] | None:
    return _load_json_dict(run_root / "analysis" / f"event_{event_index:03d}.json")


def _render_recording_event_html(*, run_root: Path, event: dict[str, Any]) -> str:
    raw_index = event.get("index")
    index = raw_index if isinstance(raw_index, int) else 0
    kind = str(event.get("kind") or "")
    kind_label = _recording_kind_label(kind)
    analysis = _load_recording_analysis(run_root, index) if index else None
    instruction = ""
    if isinstance(analysis, dict):
        raw_instruction = analysis.get("instruction")
        if isinstance(raw_instruction, str) and raw_instruction.strip():
            instruction = raw_instruction.strip()

    title = instruction or kind_label
    step_label = escape(f"{index}.")
    kind_badge = escape(kind_label)
    time_text = escape(_timestamp_text(event.get("timestamp_utc"))) or "—"

    meta_rows: list[tuple[str, str]] = [("時間", time_text)]
    cursor = _format_xy(event.get("cursor_xy"))
    if cursor:
        meta_rows.append(("游標", escape(cursor)))
    end_xy = _format_xy(event.get("end_xy"))
    if end_xy:
        meta_rows.append(("終點", escape(end_xy)))
    text = event.get("text")
    if isinstance(text, str) and text:
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
    window_title = event.get("target_window_title")
    if isinstance(window_title, str) and window_title.strip():
        meta_rows.append(("視窗", escape(window_title.strip())))

    meta_html = "".join(f"<dt>{escape(label)}</dt><dd>{value}</dd>" for label, value in meta_rows)

    shot = _resolve_recording_screenshot(str(event.get("screenshot_path") or ""), run_root)
    end_shot = _resolve_recording_screenshot(str(event.get("end_screenshot_path") or ""), run_root)
    if kind == "drag" or end_shot is not None:
        shots = _render_shot_html("開始截圖", shot, run_root) + _render_shot_html(
            "結束截圖", end_shot, run_root
        )
    else:
        shots = _render_shot_html("截圖", shot, run_root)

    return (
        f'<details class="instruction-group">'
        f"<summary>"
        f'<span class="instruction-number">{step_label}</span>'
        f'<span class="instruction-title">{escape(title)}</span>'
        f'<span class="badge neutral">{kind_badge}</span>'
        f"</summary>"
        f'<div class="meta" style="padding: 1rem 1.5rem 0;"><dl>{meta_html}</dl></div>'
        f'<div class="shots" style="padding: 0 1.5rem 1.25rem;">{shots}</div>'
        f"</details>"
    )


def _render_recording_index_row(run_root: Path) -> str:
    run_id = run_root.name
    href = escape(f"{run_id}/{_RECORDING_HTML_NAME}", quote=True)
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


def _render_runs_tab_panel(run_dirs: list[Path]) -> str:
    if run_dirs:
        rows = "".join(_render_index_row(run_dir) for run_dir in run_dirs)
        intro = (
            f"共 {len(run_dirs)} 筆報告。勾選多筆後可批次回報或刪除；"
            "點選執行名稱開啟步驟紀錄；"
            "點選欄位標題可排序；🐛 可回報 bug；"
            "垃圾桶可刪除整份報告資料夾。"
        )
        body = (
            f'<p class="intro">{intro}</p>\n'
            f"{_render_bulk_bar()}"
            '<table class="reports">'
            "<thead><tr>"
            '<th class="no-sort select-col" title="全選">'
            '<input type="checkbox" class="select-all" aria-label="全選報告">'
            "</th>"
            '<th data-type="text">執行</th>'
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
            '<p class="intro">共 0 筆報告。完成一次執行後，報告會出現在此列表。</p>\n'
            '<p class="empty">尚無報告。完成一次執行後，報告會出現在此列表。</p>'
        )
    return f'<section class="tab-panel" data-tab="runs" id="tab-runs">\n{body}\n</section>'


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
        if (run_dir / _RECORDING_HTML_NAME).is_file():
            continue
        write_recording_html_from_run(run_dir, update_index=False)


def write_runs_index_html(runs_root: Path) -> Path:
    """Build ``index.html`` with tabs for agent runs and recordings."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    _backfill_recording_html(runs_root)

    run_dirs = _iter_report_run_dirs(runs_root)
    recording_dirs = _iter_recording_report_dirs(runs_root)
    tabs = (
        '<nav class="tabs" role="tablist" aria-label="報告類型">'
        '<button type="button" class="active" data-tab="runs" role="tab" aria-selected="true"'
        ' aria-controls="tab-runs">執行報告</button>'
        '<button type="button" data-tab="recordings" role="tab" aria-selected="false"'
        ' aria-controls="tab-recordings">錄製紀錄</button>'
        "</nav>"
    )
    body = (
        f"{tabs}\n"
        f"{_render_runs_tab_panel(run_dirs)}\n"
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
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-Hant">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{_STYLE}\n</style>\n"
        "</head>\n<body>\n"
        '<p class="nav"><a href="../index.html">← 報告列表</a></p>\n'
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
        _render_recording_event_html(run_root=run_root, event=event) for event in events
    ]
    title = escape(_resolve_recording_title(run_root))
    body = "\n".join(events_html) if events_html else '<p class="empty">尚無錄製事件。</p>'
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
        f"{body}\n"
        "</body>\n</html>\n"
    )

    path = recording_html_path(run_root)
    path.write_text(html, encoding="utf-8")
    if update_index:
        write_runs_index_html(run_root.parent)
    return path
