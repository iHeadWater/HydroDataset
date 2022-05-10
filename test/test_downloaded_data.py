import os
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio.features as rio_features
import pygeoutils as geoutils
import definitions

from hydrodataset.data.data_camels import Camels
from hydrodataset.data.data_gages import Gages

from hydrodataset.utils.hydro_geo import (
    gage_intersect_time_zone,
    split_shp_to_shps_in_time_zones,
)
from hydrodataset.utils.hydro_utils import serialize_json, unserialize_json_ordered


@pytest.fixture()
def save_dir():
    dir_ = os.path.join(definitions.ROOT_DIR, "test", "test_data")
    if not os.path.isdir(dir_):
        os.makedirs(dir_)
    return dir_


@pytest.fixture()
def var():
    return ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]


@pytest.fixture()
def camels():
    camels_dir = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
    if not os.path.isfile(
        os.path.join(
            camels_dir,
            "camels_attributes_v2.0",
            "camels_attributes_v2.0",
            "camels_name.txt",
        )
    ):
        return Camels(camels_dir, True)
    return Camels(camels_dir, False)


@pytest.fixture()
def gages():
    gages_dir = os.path.join(definitions.DATASET_DIR, "gages")
    if not os.path.isfile(
        os.path.join(
            gages_dir,
            "basinchar_and_report_sept_2011",
            "spreadsheets-in-csv-format",
            "conterm_basinid.txt",
        )
    ):
        return Gages(gages_dir, True)
    return Gages(gages_dir, False)


def test1_trans_to_csv_load_to_gis(save_dir):
    basin_id = "01013500"
    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    daily = xr.open_dataset(read_path)

    arr_lat = daily["lat"].values.flatten()
    arr_lon = daily["lon"].values.flatten()
    arr_data = daily["prcp"].values[0, :, :].flatten()

    arr_all = np.c_[arr_lat, arr_lon, arr_data]
    # remove the rows with nan value
    arr = arr_all[~np.isnan(arr_all).any(axis=1)]
    df = pd.DataFrame(data=arr, columns=["lat", "lon", "prcp"])
    df.to_csv(os.path.join(save_dir, "load_to_qgis.csv"), index=False)
    # after getting the csv file, please use "Layer -> Add Layer -> Add Delimited Text Layer" in QGIS to import it.


def test2_which_basin_boundary_out_of_camels(camels, save_dir):
    basin_id = "01013500"
    camels_shp_file = camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = gpd.read_file(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
    geometry = camels_shp_epsg4326[
        camels_shp_epsg4326["hru_id"] == int(basin_id)
    ].geometry.item()
    gb = geometry.bounds
    gb_west = gb[0]
    gb_south = gb[1]
    gb_east = gb[2]
    gb_north = gb[3]

    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    daily = xr.open_dataset(read_path)

    arr_lat = daily["lat"].values.flatten()
    arr_lon = daily["lon"].values.flatten()
    arr_data = daily["prcp"].values[0, :, :].flatten()

    arr_all = np.c_[arr_lat, arr_lon, arr_data]
    # remove the rows with nan value
    arr = arr_all[~np.isnan(arr_all).any(axis=1)]
    df = pd.DataFrame(data=arr, columns=["lat", "lon", "prcp"])

    df_east = df["lon"].max()
    df_west = df["lon"].min()
    df_north = df["lat"].max()
    df_south = df["lat"].min()
    # if boundary is in the
    assert not (gb_west > df_west)
    assert not (gb_east < df_east)
    assert gb_north < df_north
    assert not (gb_south > df_south)


def test3_trans_to_rectangle(camels, save_dir):
    basin_id = "01013500"
    camels_shp_file = camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
    camels_shp = gpd.read_file(camels_shp_file)
    # transform the geographic coordinates to wgs84 i.e. epsg4326  it seems NAD83 is equal to WGS1984 in geopandas
    camels_shp_epsg4326 = camels_shp.to_crs(epsg=4326)
    geometry = camels_shp_epsg4326[
        camels_shp_epsg4326["hru_id"] == int(basin_id)
    ].geometry.item()
    save_path = os.path.join(save_dir, basin_id + "_camels.shp")
    camels_shp_epsg4326[
        camels_shp_epsg4326["hru_id"] == int(basin_id)
    ].geometry.to_file(save_path)

    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_from_urls.nc")
    ds = xr.open_dataset(read_path)
    ds_dims = ("y", "x")
    transform, width, height = geoutils.pygeoutils._get_transform(ds, ds_dims)
    _geometry = geoutils.pygeoutils._geo2polygon(geometry, "epsg:4326", ds.crs)

    _mask = rio_features.geometry_mask(
        [_geometry], (height, width), transform, invert=True
    )
    # x - column, y - row
    y_idx, x_idx = np.where(_mask)
    y_idx_min = y_idx.min()
    y_idx_max = y_idx.max()
    x_idx_min = x_idx.min()
    x_idx_max = x_idx.max()
    _mask_bound = np.full(_mask.shape, False)
    _mask_bound[y_idx_min : y_idx_max + 1, x_idx_min : x_idx_max + 1] = True

    coords = {ds_dims[0]: ds.coords[ds_dims[0]], ds_dims[1]: ds.coords[ds_dims[1]]}
    mask = xr.DataArray(_mask, coords, dims=ds_dims)
    mask_bound = xr.DataArray(_mask_bound, coords, dims=ds_dims)

    ds_masked = ds.where(mask, drop=True)
    ds_masked.attrs["transform"] = transform
    ds_masked.attrs["bounds"] = _geometry.bounds

    ds_bound_masked = ds.where(mask_bound, drop=True)
    ds_bound_masked.attrs["transform"] = transform
    ds_bound_masked.attrs["bounds"] = _geometry.bounds

    arr_lat = ds_masked["lat"].values.flatten()
    arr_lon = ds_masked["lon"].values.flatten()
    arr_data = ds_masked["prcp"].values[0, :, :].flatten()

    arr_all = np.c_[arr_lat, arr_lon, arr_data]
    # remove the rows with nan value
    arr = arr_all[~np.isnan(arr_all).any(axis=1)]
    df = pd.DataFrame(data=arr, columns=["lat", "lon", "prcp"])
    df.to_csv(os.path.join(save_dir, "geometry_load_to_qgis.csv"), index=False)

    arr_bound_lat = ds_bound_masked["lat"].values.flatten()
    arr_bound_lon = ds_bound_masked["lon"].values.flatten()
    arr_bound_data = ds_bound_masked["prcp"].values[0, :, :].flatten()

    arr_bound_all = np.c_[arr_bound_lat, arr_bound_lon, arr_bound_data]
    # remove the rows with nan value
    arr_bound = arr_bound_all[~np.isnan(arr_bound_all).any(axis=1)]
    df_bound = pd.DataFrame(data=arr_bound, columns=["lat", "lon", "prcp"])
    df_bound.to_csv(os.path.join(save_dir, "bound_load_to_qgis.csv"), index=False)
    # after getting the csv file, please use "Layer -> Add Layer -> Add Delimited Text Layer" in QGIS to import it.


def test_read_nldas_nc():
    nc_file = os.path.join("example_data", "NLDAS_FORA0125_H.A19790101.1300.020.nc")
    nc4_file = os.path.join(
        "example_data", "NLDAS_FORA0125_H.A19790101.1300.002.grb.SUB.nc4"
    )
    ds1 = xr.open_dataset(nc_file)
    ds2 = xr.open_dataset(nc4_file)
    # data from v002 and v2.0 are same
    np.testing.assert_array_equal(
        np.nansum(ds1["CAPE"].values), np.nansum(ds2["CAPE"].values)
    )


def test_read_era5_land_nc():
    nc_file = os.path.join(
        "example_data", "ERA5_LAND_20010101_20010102_total_precipitation.nc"
    )
    # nc_file = os.path.join("test_data", "a_test_range.nc")
    ds = xr.open_dataset(nc_file)
    print(ds)


def test_time_zone_gages_intersect(gages):
    gages_points_shp_file = gages.data_source_description["GAGES_POINT_SHP_FILE"]
    time_zone_shp_file = os.path.join(
        definitions.DATASET_DIR, "Time_Zones", "Time_Zones.shp"
    )
    if not os.path.isfile(time_zone_shp_file):
        raise FileNotFoundError(
            "Please download time zone file from: https://data-usdot.opendata.arcgis.com/datasets/time-zones"
        )
    gage_tz_dict = gage_intersect_time_zone(gages_points_shp_file, time_zone_shp_file)
    serialize_json(gage_tz_dict, os.path.join("test_data", "gage_tz.json"))


def test_split_shp_to_shps_in_time_zones(camels, save_dir):
    basins_shp_file = camels.data_source_description["CAMELS_BASINS_SHP_FILE"]
    gage_tz_dict = unserialize_json_ordered(os.path.join("test_data", "gage_tz.json"))
    split_shp_to_shps_in_time_zones(basins_shp_file, gage_tz_dict, save_dir)


