# Plant counting UI

Web UI and API for **object detection on georeferenced orthomosaic GeoTIFFs** (for example UAV orthos). Upload a `.tif` / `.tiff`, run tiled inference, and review counts, density, and a simple map-style view. Detections can be exported for GIS (Shapefile pipeline in the API).

**Stack:** FastAPI backend + Streamlit frontend. Inference uses an **Ultralytics-compatible YOLO** checkpoint; class labels come from the model you load.

---

## Features

- Upload georeferenced **GeoTIFF** orthomosaics.
- **Tiled inference** so large rasters are handled without loading the full image at once.
- **CRS handling**: outputs use coordinates appropriate for mapping (backend converts as needed).
- **Per-class counts**, field area / density style metrics, and a **detection heatmap** in the UI.
- Point the API at your own weights via **`MODEL_PATH`** (see below).

---

## Run locally (no Docker)

Install dependencies:

```bash
pip install -r requirements.txt
```

**Backend** (terminal 1), optional custom model:

```bash
export MODEL_PATH="/path/to/your/best.pt"
export KMP_DUPLICATE_LIB_OK=TRUE   # macOS: avoids some OpenMP conflicts
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

There is also `run_api_strawberry.sh` in the repo root as a convenience wrapper that sets `MODEL_PATH` and starts `uvicorn`.

**Frontend** (terminal 2):

```bash
cd frontend
streamlit run app.py --server.port=8501 --server.address=127.0.0.1 --server.maxUploadSize=10240
```

Ensure the frontend can reach the API (defaults expect `http://127.0.0.1:8000` unless you configure `API_BASE`).

---

## Run with Docker Compose

From the repository root:

```bash
docker compose build
docker compose up
```

- API: port **8000**
- Streamlit UI: port **8501**

Mount or bake your model under the paths expected in `docker-compose.yml`, or adjust `MODEL_PATH` and volume mappings there.
