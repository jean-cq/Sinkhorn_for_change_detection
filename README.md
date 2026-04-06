# Sinkhorn for Satellite Image Change Detection

This project investigates Sinkhorn-based optimal transport for satellite image change detection using Sentinel-2 imagery. Two representations are implemented under the same optimal transport framework: a patch-based approach, where fixed image patches are compared using spatial coordinates and deep visual features, and an object-based approach, where SAM-generated segmentation masks are converted into polygons and compared using object-level spatial, appearance, and shape descriptors. The project also includes evaluation scripts to compare the resulting change maps against a GeoAI baseline.

## Installation

Clone the repository, move into the project folder, and install the required packages:

```bash
pip install -r requirements.txt
