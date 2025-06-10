#!/bin/bash
gdown 18HaCuLpiX15Os5XCOqoenj2r__2cRHm7
echo "Downloaded autoencoder weights"

cd /home/vatsal/NWM/weather/earth-forecasting-transformer
python -m pip install -e .
echo "earthformer installed"