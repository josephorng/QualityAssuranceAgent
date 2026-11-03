# ComputerAgent 版本紀錄

（每次成功的 `nuitka.bat` 發行建置會在此處新增一筆說明。）

## v1.0.0.1 — 2026-11-03 （草稿／原文）

- Log why UIA replay repair is skipped or misses.
- Advance release version after successful Nuitka builds with Gemma Chinese changelogs.
- Stop false caption-icon OCR relabel on tiny tool windows.
- Keep click vision prefetch cache valid after settle focus_rect patches.
- Pick text clicks by constellation radius then similarity.
- Cap Triton infer concurrency and skip wasted instruction LLM on YOLO failure.
- Add mid-replay single-step re-record for click gestures.
- After overlay retry wait, re-verify before re-acting when tools already succeeded.
- Ignore untitled WinForms residue in window_verify.
- Retry find_mouse_point once when the UI changes mid-locate.
- Keep window lists when after-settle UIA times out.
- Stop live-grabbing typing OCR end frames.
- Clip caret-thin typing OCR with UIA Edit/Document bounds.
- Require appeared window verify to match recorded titles.
- Allow UIA-first clicks on sparse dialogs without a landmark-count gate.
- Strip variable Windows shell status from UIA landmark labels.
- Fix UIA replay repair for inputs and scrollbar drags.
- Keep a YOLO+OCR twin on UIA steps so replay can fall back when UIA fails.
- Support UIA replay repair for input and scrollbar class targets.
- Add UIA replay repair tests for icon OCR chinese_id confirmation.
- Run OCR during YOLO+OCR capture so text anchors are not deferred away.
- Disallow miss-path UIA writes so click naming never uses a post-click tree.
- Stop press-time UIA FindAll so click naming uses a pre-click tree.
- Gate UIA recording path on unique selectability instead of ≥8 texts.
- Prefer-replace UIA scrollbars and inputs into YOLO+OCR mouse targeting.
- icon change
- Gate settle UIA cache reuse on previous-event freshness.
- Add time profiling for the recording analyzing process.
- Match move_mouse replay targets from structured args instead of hub-prose unwrap.
- Collect UIA naming trees after settle, with press-time FindAll only on cache miss.
