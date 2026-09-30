# Improvement History

Reconstructed on 2026-09-24 from the git history (366 commits, 2026-04-16 through 2026-09-24, author SMR-AIT) and the Cursor conversation record for this repo (about 650 parent chats). Commit subjects are often short; the chats are what explain why each change was made.

The project started as a multi-process desktop agent and became a record-and-replay quality-assurance tool: capture a Windows session, turn it into a script of mouse and keyboard steps, then replay those steps with YOLO, OCR, and a language model.

| Month | Commits | Main theme |
| --- | ---: | --- |
| 2026-04 | 35 | Rebuild around modules, MCP tools, and a step loop |
| 2026-05 | 42 | GUI hub, ONNX vision, Chinese UI, packaged exe |
| 2026-06 | 15 | One mouse-move path, reports, remote vLLM |
| 2026-07 | 51 | Screen recorder, nearby landmarks, queue and smart mode |
| 2026-08 | 107 | Recording as the script, verification, typing and scrollbars |
| 2026-09 | 115 | Replay cache, settle timing, window ROI, before/after shots |

## April 16–24: first shape

`6351b36` (2026-04-16) is the first commit. The next few days cleaned the repo, rebuilt the project, stopped saving every screenshot, and added MCP (`2d21557`). Memory moved from `brain.txt` to `long_term_memory.txt`. Tools, monitor settings, and an OCR viewer followed. Prompts were renamed from skills to instructions. Text entry switched to paste so multilingual input would survive. Object detection was removed, and a prompt viewer plus a step mechanism were added so each script line could carry an image and an instruction.

The working idea at this point was a script of steps, each decided by a model and executed through MCP tools.

## April 30: drop the servers

The largest architectural change of the month came from a series of chats on April 30. The request was to stop running brain and hand as servers and use in-process modules instead. Ports and `TASK_INPUT_ENV` were removed (`877635e`, `1557013`). Screenshot and monitor code moved into `src/eye/`. The old step files were deleted (`066d2ed`) and replaced by a message loop: the brain decides a tool, the hand runs it, and the messages are saved (`e406f26`).

That loop is still the coordinator’s core. Later modes (script, queue, smart, recording replay) all sit on it.

## May: make it a desktop app

May was spent making the agent usable without a developer console.

- Coordinate selection learned to match instruction text to OCR regions by similarity (`05e7100`, May 7) instead of asking the model for raw pixels.
- A runtime-command mode let the user type the next instruction before each loop (`445516f`). Window tools learned to close, minimize, and maximize a list of windows. A storage folder held pasted text as files, not one JSON blob.
- Local PyTorch weights were replaced by ONNX, then by an end-to-end YOLO model, and Torch was removed (`aa9ebc8`, `8ae59d5`, May 14). `nuitka.bat` appeared the same day.
- `cb18455` (May 14) created the GUI. The following day laid out the main window, added stop, clear, monitor selection, and automatic focus on the command box. The UI language switched to Chinese (`e7f1073`).
- Late May simplified configuration: unused constants were removed, `eye_vlm` was dropped, and JSON config files were collapsed into code (`cb246f0`, `fe484fd`). A splash image was added for the packaged exe.

Chats in this period kept circling the same failures: hotkeys that did not fire, paste that inserted the previous clipboard, the last script step never running, and OCR that was too slow on CPU. The fixes were smaller tools and a shared LLM client that could point at a remote OpenAI-compatible server (`321363c`, May 13), which later became the vLLM host.

## June: one way to move the mouse

June is the shortest month in git (15 commits) and almost absent from dated chats, but the commits change the vision path.

A UI-element recognizer was folded into the existing OCR flow (`29f8031`). Screenshots started including the mouse cursor (`130c788`). A report mechanism was added (`94002e8`). The separate “move to text” and “move to UI element” tools became one `move_mouse` (`5287981`, June 16), then the mouse-move logic was unified again (`e9edb4d`). A cache of successful tool calls was added (`c65033f`) so a repeated script line could skip the model. `eae2588` (July 1, committed at the edge of this work) added Triton as the vision server.

The direction is already the later product: vision on a remote inference server, one mouse tool, and a written report of what the run did.

## July: record a session instead of writing a script

`7dd1f63` (July 3) added the recorder. The next commits optimized screen capture and window detection, then added drag (`5ab1d84`). From here, most chats are about a recorded event that produced the wrong instruction, or a replay that clicked the wrong icon.

The stability idea that stuck is the nearby landmark. A click is not “the folder icon”; it is “the folder icon, with Edge and Copilot beside it” (`4c45d07`, July 8). Replay stopped asking the model to invent coordinates and started matching that description against OCR and YOLO candidates. Through July 13–15 the matching was rewritten several times because small models dropped the parenthetical hint, picked a duplicate label, or ignored a pixel offset. Landmarks were reduced to the single closest anchor (`fec94bc`), then expanded again to one candidate on each side. Click and double-click were distinguished (`3b6f73c`). A `find existing text or element` tool was added so a step could be conditional (`3626400`). The model’s job was narrowed toward translation and verification rather than free assistance (`2059b95`).

Around the same days the product surface grew:

- HTML session reports, an analysis button, and an index page (`fe6f037`, `19e1d2d`, `b4c06f1`).
- A queue of scripts (`eaff0a1` era; the July 17 chat asked for sequential runs).
- Pause, unsaved-script warnings, red cross on failure (`d2d1b6b`), and “save as new file” after analysis.
- 智能模式 (`9a76971`, July 30): the whole text box is one goal, and the loop is plan, act, verify. Drag in that mode uses visual mouse resolution, and the tool-call cache is disabled because a cached click is the wrong idea for an open goal (`a5a69f3`).
- Elapsed time between recorded actions becomes a wait line when the gap is long (`62f9141`).

Window bookkeeping was a repeated failure. Closing a popup host was recorded as closing File Explorer. The recorder learned to ignore that host, to prefer the innermost control under the cursor (`7562f9d`), and to describe landmarks with a side (left, right, above, below) instead of a vague “nearby” (`99a1603`).

## August: the recording is the script

August has 107 commits. The chats stop treating the recording as a way to fill a script box and start treating the recording folder as the thing you run.

Replay reliability:

- Text targets outrank icons when they overlap (`3be7588`, `d4201e2`).
- Scrollbars are fitted and created from arrow pairs instead of trusted as raw YOLO boxes (`195b47a` and the August 11 chats).
- Serial clicks on the same spot collapse into one step (`9d2ada5`). Shift+letter and Ctrl+V become typed text, not hotkeys (`9eb7809`, `097658a`). Ctrl+A is recorded whole (`37d3eea`).
- The brain is capped at 10 inner decide steps (`b84c890`) after chats about a loop that never finished, including a model that kept calling a tool named `finish`.
- Verification text is generated during analysis (`b51e1f8`), then edited in `recording_steps.html` (`2f5e068`, `b0f4737`). Before and after screenshots are shown per step (`2d6d8ed`). A final after-shot is taken before the hub window comes back (`86b7253`).
- OCR decode becomes width-aware and returns a span per character (`68c0ac8`, `dbc5bd3`), so a click inside a word can name the character.
- Users can insert a step, including a conditional step, without re-recording (`a27d880`, `b94f70d`). Queue “add recording” accepts many folders (`cebdedf`). A run started from a recording is named after that folder (`857e9fc`).
- Typing screenshots were wrong because the first character was already on screen. The capture plan became “photo on the first key, flush after the burst” (`303b819`, `72df698`). The caret and UI Automation focus locate where the typing happened (`e7b2e82`). Recorded keystrokes are preferred over a second OCR of the field (`43c8ad6`).
- Near-miss clicks count as on target by padding OCR boxes (`bf2a078`). Chinese paths broke OpenCV’s `imread`; images are decoded in a Unicode-safe way (`20d8f00`).

The hub UX in the same month: per-step green check and red cross, copy and delete on each recorded instruction, start the queue from any item (`6d38d09`), side-by-side instruction and expected-outcome editors, and a larger gutter so step 10’s checkmark is visible.

Hold-to-click was added (`31fef48`) after a chat asking whether a long left- or right-button press could be recorded. YOLO’s 300-detection cap was handled by splitting the image (`ae32baf`). Clipboard contents are preserved when the type tool runs (`aa601bf`), which had been overwriting whatever the user had copied.

## September: replay what was recorded, on the right window

September is the densest month (115 commits) and is almost entirely about making replay match the recording.

Verification was the first focus. Expected outcomes are off unless the user turns them on. Scripted steps gained a retry limit (`5daef6d`) and a verify-branch abort when recovery cannot succeed (`e920dc4`). Verify was split into a baseline-image match and then a recovery round (`e9311e8`), polled at 1×, 1.5×, and 2× the recorded settle time (`431bf6f`). Session HTML shows the live shot next to the recorded baseline (`30672f1`, `15bcb75`). Gemma is a fallback when YOLO and OCR miss the target (`8db179d`).

Vision kept being tuned against real recordings (控制面板, 軟體資產, 打開神網, 資產設備):

- Scrollbar fit, create, and merge from arrow pairs, including vertical-plus-horizontal corners, while rejecting bars that overlap text or inputs (`f54a1e2` through `8a01487`).
- Identical icons are disambiguated with a set-cover of landmarks (`637bb23`). Each confusable peer gets two landmarks (`6663cca`). Repeated on-screen labels are skipped (`ecfa7c2`).
- Tiny YOLO text boxes are also emitted as element detections so OCR can run both streams (`d2b49e8`).
- A second OCR pass retries unknown icons (`ea945cb`, `3dc2a23`).
- `yolo_ui_small` was added and became the default (`f4090b6`, `e50f1f1`). Local Triton was removed from the hub; saved configs point at the remote host (`671f0ec`).
- Overlapping text is refined by a quadtree re-detect and then split on row-profile valleys (`ac67e03`, `77dd210`). Text boxes merge only inside one line (`d868c4e`).
- Chinese IME candidate keys are folded into the typed string (`cc27731`). A text-only model chooses between recorded keystrokes and OCR for typing steps (`11a6e47`).

Replay speed and correctness then moved together:

- Analysis compiles each step’s tool call and replay always uses that cache (`584b2a5`, `0e8b0a2`). The brain no longer re-decides a recorded step.
- Settle time is measured while recording, with Fibonacci backoff between probes (`c4e3039`, `e43a528`). Replay waits that long before the next tool, and the wait shows up in the HTML time profile (`ed1cdd1`, `3223551`).
- Mouse and keyboard screenshots, clipboard reads, and full window scans were moved off the low-level hook thread so the recorder would not drop keys (`649dc9f`, `c53bdfa`, `4e75106`, `16dd3de`).
- Vision during replay stays inside the window that was under the click, including maximized windows (`032ef55`, `ee73478`). A click on a confirmed caption button goes straight to that button (`7f5ead0`). If the window selector finds nothing, the action fails instead of clicking somewhere else (`9be1905`). If a recorded mouse target has no similarity match, cache replay fails (`66efac6`).
- Before-shots are latched from the frame before mouse-down (`6d22f45`). After-shots come from the settle probe (`c896ef4`). The hub is hidden before the final baseline and restored right after (`c44e532`, `7c2e5fe`).
- Recording steps HTML gained a playback tab (`ec4f5b8`), per-step verification checkboxes (`f818bf4`), and a time-profile tab (`56e200b`). Steps whose YOLO or OCR timed out are retried at the end of analysis, and the UI reports how many still have no target (`d6af3e1`).

The September 23–24 chats are still on the same problem: an after-shot that is really the next step’s before-shot, a dropdown that never appears in the recorded frame, and a click that did not close the menu. `window_snapshot.py` is the in-progress work for recording which window an action belongs to, so replay can check that predicate.

## What the project is now

`README.md` describes the current shape: a GUI hub (`main.py` → `app_main_hub.py`) with Eye (screenshots), Brain (script, queue, smart, and verify via vLLM), and Hand (MCP tools). Vision is a remote Triton server running YOLO and OCR. A recording folder under `recordings/` is a runnable script. A run under `runs/` keeps the step log, screenshots, and HTML report.

The improvement line, read from the commits and the chats together, is:

1. Replace servers and free-form plans with a step loop.
2. Replace local Torch with ONNX, then Triton, and ship a Chinese GUI.
3. Replace handwritten scripts with a recording.
4. Replace model-chosen pixels with landmarks, character spans, and compiled tool calls.
5. Replace “click and hope” with baseline verify, measured settle time, and a window ROI.

## Sources

- Git: `git log` from `6351b36` (2026-04-16, “first commit”) through `c896ef4` (2026-09-24, settle-probe screenshots and window-verify predicates). Counts above are commits whose dates fall in each month.
- Conversations: first user request of each parent Cursor chat for this workspace, dated when the transcript included a timestamp (about 490 of 648). Subagent transcripts were not counted separately. Undated early chats line up with the April 30 server-removal work and the early May tool fixes.
