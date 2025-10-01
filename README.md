# Weed Detection and Geolocation App

1)A **Streamlit**-based web application for detecting and geolocating weeds in **georeferenced orthomosaic GeoTIFF images** (e.g., from drone surveys).  
2)The app uses a **YOLOv8** model to process large images tile-by-tile, extract weed locations, and export a **Shapefile** with precise GPS coordinates.  
3)This tool is designed for **precision agriculture**, enabling farmers and researchers to map weeds for **spot spraying** and **field analysis**.

Demo video can be found here: https://drive.google.com/file/d/1AvmWi4ZA1eJ7l6Ai5b9v9w3ow7zuIMDL/view?usp=sharing

---

## Features
- **Upload** `.tif` / `.tiff` georeferenced orthomosaic imagery.
- **YOLOv8-based detection** on tiled images for large raster handling.
- **Automatic GPS conversion** from UTM to WGS84.
- **Export Shapefile** of detections for use in GIS software.
- **Interactive visualization** overlaying detections on the orthomosaic and important KPIs like weed density etc.
- Currently supports **grass weeds** only (common ragweed, Palmer amaranth, and common lambsquarters coming soon).

---
## Run without docker
Clone the repo and install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
Then run frontend

```bash
cd frontend
streamlit run app.py --server.port=8080 --server.address=0.0.0.0 --server.maxUploadSize=10240
```

Then run backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Running with Docker
Build an image
```bash
docker compose build --no-cache weed-geo 
```

Run 
```bash
docker compose up weed-geo
```

