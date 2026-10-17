#!/usr/bin/env bash
# Open the styled Uljin wildfire change-detection project in QGIS.
DIR="$(cd "$(dirname "$0")" && pwd)"
exec qgis --project "$DIR/uljin.qgz"
