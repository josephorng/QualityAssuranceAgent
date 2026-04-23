from __future__ import annotations

import asyncio
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
import mss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from skimage.metrics import structural_similarity

from src.common.models import EyeEvent
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
from src.common.run_state import get_run_state_manager, ts_name
from src.common.runtime_context import get_runtime_env
from src.common.monitor_prompt import read_eye_monitor_index_from_env
from src.common.settings import load_settings

settings = load_settings()
run_root, task_input, _run_id = get_runtime_env()
manager = get_run_state_manager()
ollama = OllamaClient(settings.ollama_host)
manager.init_run(task_input, run_root.name)
_last_image: np.ndarray | None = None
_first_sent = False
_active_monitor_index = read_eye_monitor_index_from_env(1)

print(
    f"[eye] init run_root={run_root} run_id={_run_id} "
    f"ports eye={settings.eye_port} brain={settings.brain_port} hand={settings.hand_port} "
    f"ollama={settings.ollama_host} vlm={settings.eye_vlm} "
    f"capture_monitor_index={_active_monitor_index}"
)
print(f"[eye] task: {task_input[:200]}{'…' if len(task_input) > 200 else ''}")
manager.log_info(
    f"Eye server initialized run_id={_run_id} "
    f"ports eye={settings.eye_port} brain={settings.brain_port} hand={settings.hand_port}"
)


class MonitorSelection(BaseModel):
    monitor_index: int


def _monitor_details() -> list[dict[str, int | str]]:
    with mss.mss() as sct:
        details: list[dict[str, int | str]] = []
        # mss index 0 is the virtual bounding monitor (all screens).
        for idx in range(len(sct.monitors)):
            monitor = sct.monitors[idx]
            entry: dict[str, int | str] = {
                "index": idx,
                "left": int(monitor["left"]),
                "top": int(monitor["top"]),
                "width": int(monitor["width"]),
                "height": int(monitor["height"]),
            }
            if idx == 0:
                entry["name"] = "all_screens"
            details.append(entry)
        return details


def _resolve_monitor_index(requested_index: int) -> int:
    with mss.mss() as sct:
        max_index = len(sct.monitors) - 1
        if max_index < 0:
            return 0
        if requested_index < 0:
            return 0
        if requested_index > max_index:
            return max_index
        return requested_index


def _grab_screenshot() -> tuple[np.ndarray, Image.Image]:
    global _active_monitor_index
    with mss.mss() as sct:
        _active_monitor_index = _resolve_monitor_index(_active_monitor_index)
        monitor = sct.monitors[_active_monitor_index]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        manager.log_info(
            f"Eye grabbed screenshot monitor={_active_monitor_index} size={img.size}"
        )
        return np.array(img.convert("L")), img


def _similarity(prev: np.ndarray, curr: np.ndarray) -> float:
    min_h = min(prev.shape[0], curr.shape[0])
    min_w = min(prev.shape[1], curr.shape[1])
    p = prev[:min_h, :min_w]
    c = curr[:min_h, :min_w]
    score = float(structural_similarity(p, c))
    manager.log_info(f"Eye calculated similarity={score} crop={min_w}x{min_h}")
    return score


async def _hand_busy() -> bool:
    url = f"http://127.0.0.1:{settings.hand_port}/state"
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(url)
            r.raise_for_status()
            busy = bool(r.json().get("busy", False))
            if busy:
                manager.log_info(
                    f"Eye hand /state busy=True status={r.status_code}, sleeping before next frame"
                )
            return busy
    except Exception as exc:
        manager.log_info(f"Eye hand /state failed url={url} err={type(exc).__name__}: {exc}")
        return False


async def _should_send(prev_image_path: Path, curr_image_path: Path) -> bool:
    manager.log_info(
        f"Eye verify_change prev={prev_image_path.name} curr={curr_image_path.name} model={settings.eye_vlm}"
    )
    out, tool_calls = await ollama.generate_json(
        settings.eye_vlm,
        prompt=get_prompt("verify_change_requires_thinking"),
        fallback={"same_state": True, "reason": "fallback"},
        image_paths=[str(prev_image_path), str(curr_image_path)],
    )
    same_state = bool(out.get("same_state", True))
    reason = out.get("reason", "")
    reason_preview = (reason[:160] + "…") if isinstance(reason, str) and len(reason) > 160 else reason
    manager.log_info(
        f"Eye verify_change result same_state={same_state} reason={reason_preview!r} "
        f"tool_calls={len(tool_calls) if tool_calls else 0}"
    )
    return not same_state


async def _send_event(event: EyeEvent) -> None:
    manager.log_info(
        f"Eye posting event screenshot={event.screenshot_name} "
        f"similarity={event.similarity_to_previous}"
    )
    brain_url = f"http://127.0.0.1:{settings.brain_port}/new_event"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(brain_url, json=event.model_dump(mode="json"))
    manager.log_info(
        f"Eye posted event screenshot={event.screenshot_name} brain_status={r.status_code} url={brain_url}"
    )


async def eye_loop() -> None:
    global _last_image
    global _first_sent
    prev_sent_image_path: Path | None = None
    while True:
        if await _hand_busy():
            await asyncio.sleep(0.3)
            continue
        paths = manager.require_paths()
        curr, screenshot_img = _grab_screenshot()

        similarity = None
        changed = True
        if _last_image is not None:
            similarity = _similarity(_last_image, curr)
            changed = similarity < settings.screenshot_similarity_threshold
        _last_image = curr

        sim_dbg = f"{similarity:.6f}" if similarity is not None else "n/a"
        manager.log_info(
            f"Eye frame decision similarity={sim_dbg} threshold={settings.screenshot_similarity_threshold} "
            f"changed={changed} first_sent={_first_sent}"
        )

        if (not _first_sent and changed) or _first_sent and changed:
            image_name = f"{ts_name()}.png"
            image_path = paths.eye_dir / image_name
            temp_path = Path(tempfile.gettempdir()) / f"eye-{uuid.uuid4().hex}.png"
            keep_screenshot = False
            try:
                screenshot_img.save(temp_path)
                manager.log_info(f"Eye candidate screenshot saved temp={temp_path} name={image_name}")
                should_send = (
                    not _first_sent
                    or prev_sent_image_path is None
                    or await _should_send(prev_sent_image_path, temp_path)
                )
                if should_send:
                    manager.log_info(f"Eye should_send=True for candidate={image_name}")
                    temp_path.replace(image_path)
                    keep_screenshot = True
                    event = EyeEvent(
                        screenshot_name=image_name,
                        screenshot_path=str(image_path),
                        similarity_to_previous=similarity,
                    )
                    await _send_event(event)
                    sim_txt = f"{similarity:.4f}" if similarity is not None else "n/a"
                    print(f"[eye] -> brain {image_name} similarity={sim_txt}")
                    manager.log_info(f"Eye sent event for {image_name}")
                    prev_sent_image_path = image_path
                    _first_sent = True
                else:
                    manager.log_info(
                        f"Eye should_send=False dropped candidate={image_name} "
                        f"(first_sent={_first_sent} prev_sent={prev_sent_image_path is not None})"
                    )
            finally:
                if not keep_screenshot and temp_path.exists():
                    temp_path.unlink()
        else:
            manager.log_info("Eye skipped frame due to high similarity")
        await asyncio.sleep(settings.screenshot_interval_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"[eye] startup eye_loop interval_s={settings.screenshot_interval_seconds}")
    manager.log_info(f"Eye lifespan startup interval={settings.screenshot_interval_seconds}")
    task = asyncio.create_task(eye_loop())
    try:
        yield
    finally:
        manager.log_info("Eye lifespan shutdown")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Eye Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/capture_targets")
async def capture_targets() -> dict[str, object]:
    return {"active_monitor_index": _active_monitor_index, "monitors": _monitor_details()}


@app.post("/capture_targets")
async def set_capture_target(selection: MonitorSelection) -> dict[str, object]:
    global _active_monitor_index
    _active_monitor_index = _resolve_monitor_index(selection.monitor_index)
    manager.log_info(f"Eye switched capture monitor={_active_monitor_index}")
    return {"active_monitor_index": _active_monitor_index, "monitors": _monitor_details()}
