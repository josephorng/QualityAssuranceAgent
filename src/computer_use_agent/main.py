from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from computer_use_agent.brain.memory import BrainMemory
from computer_use_agent.brain.reasoner import BrainReasoner
from computer_use_agent.brain.thinking_log import write_thinking_record
from computer_use_agent.core.logger import build_logger
from computer_use_agent.core.session import init_run_session
from computer_use_agent.core.state import RuntimeState, WorkerStatus
from computer_use_agent.eye.capture import capture_screenshot
from computer_use_agent.eye.describe import ScreenshotDescriber
from computer_use_agent.eye.similarity import similarity_score
from computer_use_agent.hand.executor import HandExecutor
from computer_use_agent.mcp.bootstrap import build_registry
from computer_use_agent.storage.manager import StorageManager


def load_constants(project_root: Path) -> dict:
    path = project_root / "src" / "computer_use_agent" / "config" / "constants.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run(task: str, dry_run: bool = True, max_cycles_override: int | None = None) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    constants = load_constants(project_root)
    run_paths = init_run_session(project_root=project_root, task=task)
    logger = build_logger(run_paths.log_file)
    logger.info("run_started task=%s run_dir=%s", task, run_paths.root)

    state = RuntimeState()
    describer = ScreenshotDescriber()
    memory = BrainMemory(run_paths.brain_txt, int(constants["brain_memory_max_chars"]))
    reasoner = BrainReasoner()
    registry = build_registry(dry_run=dry_run)
    hand = HandExecutor(registry=registry, hand_csv_path=run_paths.hand_csv)
    storage = StorageManager(
        storage_dir=run_paths.storage_dir, storage_index_path=run_paths.storage_json
    )

    previous_image = None
    max_cycles = (
        int(max_cycles_override)
        if max_cycles_override is not None
        else int(constants["max_cycles"])
    )
    interval = float(constants["screenshot_interval_seconds"])
    threshold = float(constants["similarity_threshold"])

    for cycle in range(max_cycles):
        state.eye_status = WorkerStatus.BUSY
        current_image = capture_screenshot(run_paths.eye_dir)
        state.latest_image_name = current_image.name
        state.eye_status = WorkerStatus.IDLE
        logger.info("screenshot cycle=%s name=%s", cycle, current_image.name)

        should_process = previous_image is None
        if previous_image is not None:
            score = similarity_score(current_image, previous_image)
            logger.info("similarity cycle=%s score=%.5f", cycle, score)
            should_process = score < threshold or (
                state.brain_status == WorkerStatus.IDLE
                and state.hand_status == WorkerStatus.IDLE
            )

        if not should_process:
            previous_image = current_image
            time.sleep(interval)
            continue

        if state.brain_status == WorkerStatus.BUSY:
            state.interrupted_reasoning = True

        state.brain_status = WorkerStatus.BUSY
        description = describer.describe(current_image)
        decision = reasoner.decide(
            task=task,
            screenshot_description=description,
            memory_context=memory.read(),
            cycle_index=cycle,
            max_cycles=max_cycles,
        )

        write_thinking_record(
            thinking_dir=run_paths.thinking_dir,
            image_name=current_image.stem,
            thought=decision.thought,
            decision={
                "done": decision.done,
                "tool": decision.tool,
                "args": decision.args or {},
            },
            interrupted=state.interrupted_reasoning,
        )

        memory.append(
            f"[{current_image.name}] description={description} thought={decision.thought}"
        )

        if decision.done:
            logger.info("brain_decision done=True cycle=%s", cycle)
            storage.save_text(
                name=f"{current_image.stem}-result.txt",
                content=f"Task stopped at cycle {cycle}.",
                summary="Run completion marker",
            )
            state.brain_status = WorkerStatus.IDLE
            break

        if decision.tool:
            tool_kind = registry.kind(decision.tool)
            if tool_kind == "retrieval":
                retrieval_result = registry.run(decision.tool, decision.args)
                memory.append(
                    f"[retrieval] tool={decision.tool} result={json.dumps(retrieval_result)}"
                )
                logger.info(
                    "retrieval tool=%s result=%s", decision.tool, retrieval_result
                )
            else:
                state.hand_status = WorkerStatus.BUSY
                action_result = hand.execute(
                    image_name=current_image.name,
                    tool_name=decision.tool,
                    args=decision.args,
                )
                state.hand_status = WorkerStatus.IDLE
                memory.append(
                    f"[interaction] tool={decision.tool} result={json.dumps(action_result)}"
                )
                logger.info("interaction tool=%s result=%s", decision.tool, action_result)

        state.brain_status = WorkerStatus.IDLE
        previous_image = current_image
        time.sleep(interval)

    logger.info("run_completed run_dir=%s", run_paths.root)
    return run_paths.root


def main() -> None:
    parser = argparse.ArgumentParser(description="Computer Use Agent scaffold runner.")
    parser.add_argument("task", help="Task description text to execute.")
    parser.add_argument(
        "--live-input",
        action="store_true",
        help="Enable real mouse/keyboard actions (default is dry-run).",
    )
    args = parser.parse_args()
    run_dir = run(args.task, dry_run=not args.live_input)
    print(f"Run completed. Artifacts saved at: {run_dir}")


if __name__ == "__main__":
    main()
