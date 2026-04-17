from __future__ import annotations

import asyncio
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
ollama = OllamaClient(settings.ollama_host)
run_root, task_input, _run_id = get_runtime_env()
manager = RunStateManager(run_root.parent, settings.brain_memory_max_chars)
manager.init_run(task_input, run_root.name)
_last_image: np.ndarray | None = None
_first_sent = False

print(
    f"[eye] init run_root={run_root} run_id={_run_id} "
    f"ports eye={settings.eye_port} brain={settings.brain_port} hand={settings.hand_port} "
    f"ollama={settings.ollama_host} vlm={settings.eye_vlm}"
)
print(f"[eye] task: {task_input[:200]}{'…' if len(task_input) > 200 else ''}")


def _grab_screenshot(path: Path) -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save(path)
        return np.array(img.convert("L"))


def _similarity(prev: np.ndarray, curr: np.ndarray) -> float:
    min_h = min(prev.shape[0], curr.shape[0])
    min_w = min(prev.shape[1], curr.shape[1])
    p = prev[:min_h, :min_w]
    c = curr[:min_h, :min_w]
    return float(structural_similarity(p, c))


async def _hand_busy() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(f"http://127.0.0.1:{settings.hand_port}/state")
            r.raise_for_status()
            return bool(r.json().get("busy", False))
    except Exception:
        return False


async def _describe_image(image_name: str, image_path: Path) -> str:
    prompt = get_prompt("describe_screenshot")
    full_prompt = f"{prompt}\n\nTask:\n{task_input}\n\nImage file name: {image_name}\nTimestamp: {datetime.utcnow().isoformat()}"
    return await ollama.generate(settings.eye_vlm, full_prompt, image_paths=[str(image_path)])


async def _should_send(prev_desc: str, curr_desc: str) -> bool:
    prompt = get_prompt("verify_change_requires_thinking")
    full_prompt = f"{prompt}\n\nPrevious:\n{prev_desc}\n\nCurrent:\n{curr_desc}"
    out = await ollama.generate_json(
        settings.eye_vlm,
        full_prompt,
        fallback={"requires_thinking": True, "reason": "fallback"},
    )
    return bool(out.get("requires_thinking", True))


async def _send_event(event: EyeEvent) -> None:
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"http://127.0.0.1:{settings.brain_port}/new_event",
            json=event.model_dump(mode="json"),
        )


async def eye_loop() -> None:
    global _last_image
    global _first_sent
    prev_desc = ""
    while True:
        if await _hand_busy():
            await asyncio.sleep(0.3)
            continue
        paths = manager.require_paths()
        image_name = f"{ts_name()}.png"
        image_path = paths.eye_dir / image_name
        curr = _grab_screenshot(image_path)

        similarity = None
        changed = True
        if _last_image is not None:
            similarity = _similarity(_last_image, curr)
            changed = similarity < settings.screenshot_similarity_threshold
        _last_image = curr

        if (not _first_sent and changed) or _first_sent and changed:
            desc = await _describe_image(image_name, image_path)
            if (not _first_sent) or await _should_send(prev_desc, desc):
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
                prev_desc = desc
                _first_sent = True
        await asyncio.sleep(settings.screenshot_interval_seconds)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(f"[eye] startup eye_loop interval_s={settings.screenshot_interval_seconds}")
    task = asyncio.create_task(eye_loop())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Eye Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
