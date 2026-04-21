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
from PIL import Image
from skimage.metrics import structural_similarity

from src.common.models import EyeEvent
from src.common.ollama_client import OllamaClient
from src.common.prompting import get_prompt
from src.common.run_state import RunStateManager, ts_name
from src.common.runtime_context import get_runtime_env
from src.common.settings import load_settings

settings = load_settings()
run_root, task_input, _run_id = get_runtime_env()
manager = RunStateManager(run_root.parent, settings.brain_memory_max_chars)
ollama = OllamaClient(settings.ollama_host, log_manager=manager)
manager.init_run(task_input, run_root.name)
_last_image: np.ndarray | None = None
_first_sent = False

print(
    f"[eye] init run_root={run_root} run_id={_run_id} "
    f"ports eye={settings.eye_port} brain={settings.brain_port} hand={settings.hand_port} "
    f"ollama={settings.ollama_host} vlm={settings.eye_vlm}"
)
print(f"[eye] task: {task_input[:200]}{'…' if len(task_input) > 200 else ''}")
manager.log_debug(
    f"Eye server initialized run_id={_run_id} "
    f"ports eye={settings.eye_port} brain={settings.brain_port} hand={settings.hand_port}"
)


def _grab_screenshot() -> tuple[np.ndarray, Image.Image]:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        manager.log_debug(f"Eye grabbed screenshot size={img.size}")
        return np.array(img.convert("L")), img


def _similarity(prev: np.ndarray, curr: np.ndarray) -> float:
    min_h = min(prev.shape[0], curr.shape[0])
    min_w = min(prev.shape[1], curr.shape[1])
    p = prev[:min_h, :min_w]
    c = curr[:min_h, :min_w]
    manager.log_debug(f"Eye calculated similarity={float(structural_similarity(p, c))}")
    return float(structural_similarity(p, c))


async def _hand_busy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"http://127.0.0.1:{settings.hand_port}/state")
            r.raise_for_status()
            return bool(r.json().get("busy", False))
    except Exception:
        return False


async def _describe_image(image_path: Path) -> str:
    prompt = get_prompt("describe_screenshot")
    with Image.open(image_path) as img:
        width, height = img.size
    prompt = f"{prompt}\n\nImage size: {width}x{height} pixels (width x height)."
    text, _tool_calls = await ollama.generate(settings.eye_vlm, prompt, image_paths=[str(image_path)])
    return text


async def _should_send(prev_image_path: Path, curr_image_path: Path) -> bool:
    prompt = get_prompt("verify_change_requires_thinking")
    full_prompt = (
        f"{prompt}\n\n"
        "The first image is the previous screenshot and the second image is the current screenshot."
    )
    out, _tool_calls = await ollama.generate_json(
        settings.eye_vlm,
        full_prompt,
        fallback={"requires_thinking": True, "reason": "fallback"},
        image_paths=[str(prev_image_path), str(curr_image_path)],
    )
    return bool(out.get("requires_thinking", True))


async def _send_event(event: EyeEvent) -> None:
    manager.log_debug(
        f"Eye posting event screenshot={event.screenshot_name} "
        f"similarity={event.similarity_to_previous}"
    )
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"http://127.0.0.1:{settings.brain_port}/new_event",
            json=event.model_dump(mode="json"),
        )
    manager.log_debug(f"Eye posted event screenshot={event.screenshot_name}")


async def eye_loop() -> None:
    global _last_image
    global _first_sent
    prev_sent_image_path: Path | None = None
    while True:
        if await _hand_busy():
            manager.log_debug("Eye waiting because hand is busy")
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

        if (not _first_sent and changed) or _first_sent and changed:
            image_name = f"{ts_name()}.png"
            image_path = paths.eye_dir / image_name
            temp_path = Path(tempfile.gettempdir()) / f"eye-{uuid.uuid4().hex}.png"
            keep_screenshot = False
            try:
                screenshot_img.save(temp_path)
                desc = await _describe_image(temp_path)
                should_send = (
                    not _first_sent
                    or prev_sent_image_path is None
                    or await _should_send(prev_sent_image_path, temp_path)
                )
                if should_send:
                    temp_path.replace(image_path)
                    keep_screenshot = True
                    event = EyeEvent(
                        screenshot_name=image_name,
                        screenshot_path=str(image_path),
                        description=desc,
                        similarity_to_previous=similarity,
                    )
                    await _send_event(event)
                    sim_txt = f"{similarity:.4f}" if similarity is not None else "n/a"
                    desc_preview = (desc[:120] + "…") if len(desc) > 120 else desc
                    print(f"[eye] -> brain {image_name} similarity={sim_txt} desc={desc_preview!r}")
                    manager.log_debug(f"Eye sent event for {image_name}")
                    prev_sent_image_path = image_path
                    _first_sent = True
                else:
                    manager.log_debug(f"Eye dropped screenshot={image_name} after verification")
            finally:
                if not keep_screenshot and temp_path.exists():
                    temp_path.unlink()
        else:
            manager.log_debug("Eye skipped frame due to high similarity")
        await asyncio.sleep(settings.screenshot_interval_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"[eye] startup eye_loop interval_s={settings.screenshot_interval_seconds}")
    manager.log_debug(f"Eye lifespan startup interval={settings.screenshot_interval_seconds}")
    task = asyncio.create_task(eye_loop())
    try:
        yield
    finally:
        manager.log_debug("Eye lifespan shutdown")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Eye Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
