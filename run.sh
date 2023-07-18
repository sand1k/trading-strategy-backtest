#!/bin/sh

zipline ingest -b sharadar
if [ $? -ne 0 ]; then
  echo “Error: Failed to ingest data”
  exit 1
fi
for f in strategies/*.py;
do
  echo "Running $f"
  python "$f";
done