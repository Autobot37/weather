#!/bin/bash

echo "Start"

# Set the timestamp (you might want to define it)
timestamp=$(date +%Y%m%d)

# Run the model command
ai-models --download-assets graphcast \
  --input cds \
  --date "$timestamp" \
  --time 1200 \
  --lead-time 24 \
  --output none \
  --debug

# Move the output file to your Dataserver directory
mv output.nc "/home/vatsal/Dataserver/graphcast/output_${timestamp}.nc"

echo "End"
