from __future__ import annotations

import pyautogui


def click(args: dict) -> dict:
    x = int(args.get("x", 100))
    y = int(args.get("y", 100))
    pyautogui.click(x=x, y=y)
    return {"ok": True, "action": "click", "x": x, "y": y}


def type_text(args: dict) -> dict:
    text = str(args.get("text", ""))
    if text:
        pyautogui.write(text, interval=0.01)
    return {"ok": True, "action": "type_text", "chars": len(text)}


def key_press(args: dict) -> dict:
    key = str(args.get("key", "enter"))
    pyautogui.press(key)
    return {"ok": True, "action": "key_press", "key": key}
