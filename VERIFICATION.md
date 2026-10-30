# Verification points

What a replay step can check, and where each value comes from. Window and signal checks live in `src/recorder/window_snapshot.py` and `src/recorder/verify_signals.py`. The step order lives in `BrainModule.process_step` in `src/brain/module.py`.

A missing field asserts nothing. Extra live window changes are ignored. A recorded field must match the live after-sample.

## When a step is checked

```mermaid
flowchart TD
  Act[Actor finishes the step]
  Act --> Win{Recorded window_verify has any assertion?}
  Win -->|yes, and tools succeeded| Poll[Poll window list and signals]
  Poll -->|match| Advance[advance]
  Poll -->|miss at deadline| Retry[retry, clearly_unmet]
  Win -->|no, and tools succeeded| Advance
  Win -->|tools failed| Fail[fail the step]
```

Replay does not send a screenshot to a verifier LLM. A successful actor with no recorded window or signal assertion advances without a screenshot.

`settle_after` is the recording gap until the next event. Polling starts immediately after the actions and repeats every 0.1 seconds. The deadline is 2.0× that gap when the gap is at least 2 seconds, otherwise 1.0×. `appeared` floors the budget at 5 seconds. `disappeared` and signal assertions (`foreground`, `click_window`, clipboard, caret, focused, scroll) floor it at 1 second. When `close_windows` succeeded and a disappeared window is still present, verify re-closes it until the check passes or the deadline, which is at least 10 seconds.

## Where the recorded predicate comes from

Recording stores two samples on `window_snapshot_debug`, then `window_verify_from_debug` writes `analysis/event_NNN.json` → `window_verify`. Replay loads that dict through `collect_recording_window_verifies`.

| Sample | When | What fills it |
| --- | --- | --- |
| Before | Copied from the pre-click context cache at the gesture | `screen-recorder-preclick-context`. Every 0.25 seconds: top-level windows, foreground, caret. At most once a second: clipboard, UI Automation. |
| Press-time point | On the mouse hook, at the click coordinates | `WindowFromPoint` + `GetAncestor(GA_ROOT)`. This overwrites `signals_before.point_window`. It is also the `click_window` passed to `move_mouse`. |
| After | Window-step thread, after the remaining 0.25 seconds (0.45 seconds for a title-bar click). A text step flushed by the next click or key uses the settle cache copied on the hook before that gesture, so the gesture's window change stays on the gesture. | A fresh top-level window list, then `capture_step_signals` at the same cursor. UI Automation runs only for a click, text input, or scroll, with a 1 second timeout. On timeout the window list and fast signals are kept; only UIA fields are skipped and `uia_timed_out` is set on the debug dict. The sealed text after-sample skips that live read. |

`signal_verify_fields` keeps only values that changed. The stored value is the after-sample.

## Window list

`snapshot_top_level_windows` enumerates visible top-level windows and fills each window's executable name from one process snapshot. `appeared`, `disappeared`, and `state` store that name with the class name. Recording still pairs a window with itself by hwnd, and same-identity hwnd churn uses class, normalized title, and process name. Replay of those three compares class name and process name only. Title is stored for display and is not compared, because it can differ on the next run. A blank recorded class or process matches any live value. An entry with neither does not match. The agent hub window (`電腦使用代理`), the taskbar, and the input-pane strip (`EdgeUiInputTopWndClass`) are dropped. Replay skips those classes when an older analysis still lists them. A title change on an hwnd that is still present is not an appear or disappear.

| Point | Recorded when | Replay read |
| --- | --- | --- |
| `appeared` | An hwnd is in the after list and was not in the before list. Taskbar, the input-pane strip, and the agent hub are dropped. Same-identity hwnd churn cancels out. The entry is kept only when this step's after-foreground, or a later step's foreground, has the same class name and executable name. | The window must be present in the live after list. It does not need to newly appear during the step. |
| `disappeared` | An hwnd was in the before list and is gone after. Taskbar, the input-pane strip, and the agent hub are dropped. | The window must be absent from the live after list. It does not need to have been open before the step. |
| `state` | The same hwnd is still there and `is_minimized` or `is_maximized` flipped. Label is `minimized`, `restored`, `maximized`, or `unmaximized`. | The matching window must be present in the live after list and already hold that end flag (`maximized` → `is_maximized`, `minimized` → `is_minimized`, `restored` → not minimized, `unmaximized` → not maximized). It does not need to flip during the step. |

Replay takes the before list at the start of the step and the after list after `settle_after`.

## Signals

Replay reads a signal only when the recorded predicate contains that field (`capture_replay_after_signals`). `foreground` and `click_window` match on class name and process name. Title is not compared. A blank recorded class or process matches any live value. An entry with neither does not match.

| Point | Recorded when | How the after value is read | Replay read |
| --- | --- | --- | --- |
| `foreground` | `GetForegroundWindow` changed, and the after window is not the agent hub. Omitted when `appeared`, `disappeared`, or `state` is also present (next-focus after those changes is ignored on replay too). | `GetForegroundWindow`, then class, title, and the executable name of that hwnd's process. | Same call after the step. A live agent-hub foreground fails. |
| `click_window` | Click, double-click, triple-click, right-click, or middle-click, and the root window under the cursor changed between press and settle. The stored identity is the **press-time** window, including its executable name. | `WindowFromPoint` on the mouse hook at the click, then the executable name of that process, saved as `signals_before.point_window`. | The click tool calls `window_at_point` at the click coordinates before the button goes down. Verification uses that identity and does not read the cursor again after settle. |
| `clipboard` | Clipboard text changed. | `pyperclip.paste`, giving up after 0.25 seconds. Text is cut at 4000 characters. | Same read. Compared as an exact string. |
| `caret` | Text input, and the caret window changed: a different hwnd, or the caret appeared. Movement inside the same hwnd is omitted. The field is omitted unless both the class name and the executable name were read. | `GetGUIThreadInfo` on the foreground thread for `hwndCaret`, then `GetClassNameW` and the executable file name of that window's process. The stored value is those two names. | The live caret window's class name and executable file name must both match. Screen position is not compared. |
| `focused` | Text input, and the focused element's name or value changed. A name equal to the foreground title is ignored. A value that is neither equal to the typed text nor a prefix of it (or the reverse) is omitted: an address bar can report a fragment that is not the text the step types. | UI Automation `GetFocusedElement`: current name, and the value pattern when present. | Exact `name`. `value` matches when the strings are equal, or one is a prefix of the other, so an inline-autocomplete tail on either side still passes. |
| `scroll` | A scroll, and the percent moved by more than 0.5. Stored to one decimal. | UI Automation element at the cursor, walking up to 8 parents, from the range-value pattern: `(value - min) / (max - min) * 100`. | Within 5 percent. |

`control_state` may still appear in recorded `signals_before` / `signals_after` for debugging. It is not written into `window_verify` and is ignored if an older analysis still has it.

Recorded baseline frames and `expected_outcome` text stay on the report. Replay does not compare them to a live screenshot.
