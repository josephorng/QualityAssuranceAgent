# NVIDIA Triton — YOLO + CRNN OCR

將 `cua_mcp/best.onnx` 以 **yolo_ui** 提供服務，將 `cua_mcp/read_screen_text/ocr_model_finetuned.onnx` 以 **crnn_ocr** 提供服務。

前處理（letterbox、裁切縮放）與後處理（框解碼、CTC 解碼）仍由 Python 用戶端執行。

## 先決條件

- Docker
- GPU 推論：NVIDIA 驅動程式 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

若未啟用 GPU 直通，Triton 仍會透過 CPU `instance_group` 提供兩個模型的服務。

## 連接埠

本專案僅使用 HTTP 推論，因此 `docker-compose.yml` 只將主機 **9000** 對應到容器內 Triton HTTP 埠 **8000**（避開本機已被佔用的 8000）。

| 主機連接埠 | 容器埠 | 協定 | 用途 |
|------------|--------|------|------|
| **9000** | 8000 | HTTP/REST | 推論 API（健康檢查、模型就緒、推論請求）。Python 用戶端與 `TRITON_HTTP_URL` 使用此埠。 |

gRPC（容器 8001）與 Prometheus 指標（容器 8002）未對外映射；若日後需要，再於 `docker-compose.yml` 加入對應埠即可。

## 快速開始

在儲存庫根目錄執行：

```bash
python triton/scripts/sync_models.py
docker compose -f triton/docker-compose.yml up
```

健康檢查：

```bash
curl http://localhost:9000/v2/health/ready
curl http://localhost:9000/v2/models/yolo_ui/ready
curl http://localhost:9000/v2/models/crnn_ocr/ready
```

## GPU 驗證

啟用 GPU 直通後：

```bash
nvidia-smi
curl http://localhost:9000/v2/models/yolo_ui/stats
```

## 僅 CPU 模式

移除或註解 `docker-compose.yml` 中的 `deploy.resources` GPU 區塊，然後：

```bash
docker compose -f triton/docker-compose.yml up
curl http://localhost:9000/v2/models/yolo_ui/ready
```

模型會透過 CPU instance group 維持 `READY`。

## Python 用戶端

在 `.env` 中設定：

```
TRITON_HTTP_URL=http://localhost:9000
VISION_BACKEND=auto
```

- `auto` — Triton 可連線時使用 Triton；伺服器離線時改用本機 ONNX Runtime（僅開發用）。
- `triton` — 必須使用 Triton（不可用則失敗）。
- `local` — 一律使用內建／本機 ONNX Runtime。

Nuitka 建置的 `ComputerAgent.exe` **不會**打包 ONNX 權重；正式環境需將 `TRITON_HTTP_URL` 指向 Triton 主機。
