"""Step 8 (final goal) — build a ready-to-open QGIS project (.qgz).

Assembles the outputs into one styled project so opening it in QGIS shows, over a
post-fire false-colour basemap:
    - Burn perimeter (GeoJSON polygons, red)
    - Land-cover transition (from->to, categorical colours)   [on]
    - Burn severity (dNBR classes)                            [off by default]
Run with SYSTEM python (has PyQGIS + numpy + gdal), headless:
    QT_QPA_PLATFORM=offscreen python3 src/make_qgis.py
Then:  qgis --project uljin.qgz   (or run open_qgis.sh)
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from osgeo import gdal
from qgis.core import (QgsApplication, QgsProject, QgsRasterLayer, QgsVectorLayer,
                       QgsPalettedRasterRenderer, QgsFillSymbol)
from qgis.PyQt.QtGui import QColor
gdal.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "outputs")
CD = os.path.join(HERE, "..", "data", "cd")
PROJ = os.path.abspath(os.path.join(HERE, "..", "uljin.qgz"))
BASEMAP = os.path.join(OUT, "post_rgb.tif")


CLOUDMASK = os.path.join(OUT, "cloudmask.tif")


def build_basemap():
    d = np.load(os.path.join(CD, "uljin_full.npz"), allow_pickle=True)
    img = d["image"]; gt = tuple(d["geotransform"].tolist()); wkt = str(d["projection"])
    valid = d["valid"].astype(np.uint8)
    H, W = img.shape[1], img.shape[2]
    ds = gdal.GetDriverByName("GTiff").Create(BASEMAP, W, H, 3, gdal.GDT_Byte,
                                              ["COMPRESS=DEFLATE", "TILED=YES", "PHOTOMETRIC=RGB"])
    ds.SetGeoTransform(gt); ds.SetProjection(wkt)
    for i, ch in enumerate([6, 7, 8]):     # post fR=B12, fG=B08, fB=B02
        ds.GetRasterBand(i+1).WriteArray((img[ch]*255).astype(np.uint8))
    ds = None
    # explicit cloud/invalid mask (SCL-masked pixels: cloud/shadow/cirrus/snow/nodata) = 1
    cm = (1 - valid).astype(np.uint8)
    cds = gdal.GetDriverByName("GTiff").Create(CLOUDMASK, W, H, 1, gdal.GDT_Byte,
                                               ["COMPRESS=DEFLATE", "TILED=YES"])
    cds.SetGeoTransform(gt); cds.SetProjection(wkt); cds.GetRasterBand(1).WriteArray(cm); cds = None


def paletted(path, name, classes):
    lyr = QgsRasterLayer(path, name)
    cs = [QgsPalettedRasterRenderer.Class(v, QColor(*rgb), lab) for v, rgb, lab in classes]
    lyr.setRenderer(QgsPalettedRasterRenderer(lyr.dataProvider(), 1, cs))
    return lyr


def main():
    build_basemap()
    QgsApplication.setPrefixPath("/usr", True)
    app = QgsApplication([], False); app.initQgis()
    project = QgsProject.instance()

    base = QgsRasterLayer(BASEMAP, "Post 2022-04-09 (false colour B12/B08/B02)")
    project.setCrs(base.crs())
    project.addMapLayer(base)

    sev = paletted(os.path.join(OUT, "severity_pred.tif"), "Burn severity (dNBR)",
                   [(1, (255, 255, 80), "Low"), (2, (255, 165, 0), "Moderate-low"),
                    (3, (255, 70, 0), "Moderate-high"), (4, (180, 0, 0), "High")])
    project.addMapLayer(sev)

    trans = paletted(os.path.join(OUT, "transition.tif"), "Land-cover transition (from->to)",
                     [(1, (220, 0, 0), "Veg -> Burned scar"), (2, (255, 140, 0), "Healthy -> Sparse veg"),
                      (3, (220, 220, 0), "Veg -> Bare soil"), (4, (150, 150, 150), "Cloud / invalid")])
    project.addMapLayer(trans)

    cloud = paletted(CLOUDMASK, "Cloud / invalid mask (SCL)", [(1, (0, 200, 255), "Cloud / invalid")])
    project.addMapLayer(cloud)

    burn = QgsVectorLayer(os.path.join(OUT, "burn_final.geojson"), "Burn perimeter", "ogr")
    burn.renderer().setSymbol(QgsFillSymbol.createSimple(
        {"color": "255,0,0,60", "outline_color": "255,0,0,255", "outline_width": "0.4"}))
    project.addMapLayer(burn)

    # severity + cloud-mask off by default (transition + perimeter shown)
    root = project.layerTreeRoot()
    root.findLayer(sev.id()).setItemVisibilityChecked(False)
    root.findLayer(cloud.id()).setItemVisibilityChecked(False)

    print("saved:", PROJ, "| layers:", [l.name() for l in project.mapLayers().values()])
    print("CRS:", project.crs().authid())
    project.write(PROJ)
    app.exitQgis()


if __name__ == "__main__":
    main()
