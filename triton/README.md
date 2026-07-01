# NVIDIA Triton — YOLO + CRNN OCR

Serves `cua_mcp/best.onnx` as **yolo_ui** and `cua_mcp/read_screen_text/ocr_model_finetuned.onnx` as **crnn_ocr**.

Preprocessing (letterbox, crop resize) and postprocessing (box decode, CTC decode) stay in the Python client.

## Prerequisites

- Docker
- For GPU inference: NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Without GPU passthrough, Triton still serves both models on the CPU `instance_group`.

## Quick start

From the repository root:

```bash
python triton/scripts/sync_models.py
docker compose -f triton/docker-compose.yml up
```

Health check:

```bash
curl http://localhost:8000/v2/health/ready
curl http://localhost:8000/v2/models/yolo_ui/ready
curl http://localhost:8000/v2/models/crnn_ocr/ready
```

## GPU verification

With GPU passthrough enabled:

```bash
nvidia-smi
curl http://localhost:8000/v2/models/yolo_ui/stats
```

## CPU-only mode

Remove or comment out the `deploy.resources` GPU block in `docker-compose.yml`, then:

```bash
docker compose -f triton/docker-compose.yml up
curl http://localhost:8000/v2/models/yolo_ui/ready
```

Models remain `READY` via the CPU instance group.

## Python client

Set in `.env`:

```
TRITON_HTTP_URL=http://localhost:8000
VISION_BACKEND=auto
```

- `auto` — use Triton when reachable; fall back to local ONNX Runtime when the server is down (dev only).
- `triton` — require Triton (fail if unavailable).
- `local` — always use bundled/local ONNX Runtime.

The Nuitka `ComputerAgent.exe` build does **not** bundle ONNX weights; production expects `TRITON_HTTP_URL` pointing at a Triton host.
