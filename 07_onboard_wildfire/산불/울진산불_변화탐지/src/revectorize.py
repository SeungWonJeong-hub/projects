"""Vectorize burn_final.tif -> GeoJSON of burn polygons only (val==1), with area_ha.
Polygonize into an in-memory OGR layer first, then copy burn features out.

## Area is measured in UTM, coordinates are written in WGS84

GeoJSON (RFC 7946) allows **only** EPSG:4326 lon/lat. An earlier version wrote the
polygons straight out in the source UTM 52N (EPSG:32652): QGIS still opened it,
because the file declares its CRS, but anything that follows the spec — web maps,
`geojson.io`, most JS libraries — placed the burn scar in the Atlantic, since it
read 511533, 4090346 as degrees.

Areas must NOT be measured after that transform: a degree of longitude is ~0.8x a
degree of latitude at 37 deg N, so `GetArea()` on lon/lat is meaningless. So the
order matters — **measure first in UTM, transform second** — and `area_ha` rides
along as an attribute. This is the same rule `s2_water_cd_app/core/vectorize.py`
follows, and the one `app/segdemo/export_gis.py` states in its header.
"""
import os
import numpy as np
from osgeo import gdal, ogr, osr
gdal.UseExceptions()

HERE = os.path.dirname(__file__)
OUTTIF = os.path.join(HERE, "..", "data", "outputs")
src = gdal.Open(os.path.join(OUTTIF, "burn_final.tif"))
arr = (src.GetRasterBand(1).ReadAsArray() > 0).astype(np.uint8)
print("burn pixels:", int(arr.sum()))

mem = gdal.GetDriverByName("MEM").Create("", arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte)
mem.SetGeoTransform(src.GetGeoTransform()); mem.SetProjection(src.GetProjection())
mem.GetRasterBand(1).WriteArray(arr)
mb = mem.GetRasterBand(1)

srs = osr.SpatialReference(); srs.ImportFromWkt(src.GetProjection())
memds = ogr.GetDriverByName("Memory").CreateDataSource("m")
mlyr = memds.CreateLayer("poly", srs=srs, geom_type=ogr.wkbPolygon)
mlyr.CreateField(ogr.FieldDefn("val", ogr.OFTInteger))
gdal.Polygonize(mb, mb, mlyr, 0, [])   # mask=mb: only burn(=1) pixels

wgs84 = osr.SpatialReference()
wgs84.ImportFromEPSG(4326)
# GDAL 3 orders axes by the authority definition, so EPSG:4326 would come out
# lat,lon. GeoJSON wants lon,lat -> say so explicitly rather than swapping later.
wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
to_wgs84 = osr.CoordinateTransformation(srs, wgs84)

out = os.path.join(OUTTIF, "burn_final.geojson")
if os.path.exists(out): os.remove(out)
ods = ogr.GetDriverByName("GeoJSON").CreateDataSource(out)
olyr = ods.CreateLayer("burn", srs=wgs84, geom_type=ogr.wkbPolygon)
olyr.CreateField(ogr.FieldDefn("area_ha", ogr.OFTReal))
defn = olyr.GetLayerDefn()

mlyr.ResetReading()
tot = 0.0; n = 0
for feat in mlyr:
    if feat.GetField("val") != 1:
        continue
    geom = feat.GetGeometryRef().Clone()
    a = geom.GetArea() / 1e4          # UTM metres -> ha. MUST be before the transform.
    geom.Transform(to_wgs84)
    of = ogr.Feature(defn); of.SetGeometry(geom); of.SetField("area_ha", round(a, 2))
    olyr.CreateFeature(of); tot += a; n += 1
ods = None; memds = None; mem = None
print(f"polygons: {n}, total {tot:.0f} ha -> {out}")
