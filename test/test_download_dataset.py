import os
import pytest
import geopandas as gpd
from pynhd import NLDI
import xarray as xr
import pydaymet as daymet

import definitions

from hydrodataset.data.data_camels import Camels
from hydrodataset.data.data_gages import read_usgs_daily_flow

from hydrodataset.utils.hydro_utils import unserialize_geopandas



@pytest.fixture()
def save_dir():
    save_dir_ = os.path.join(definitions.ROOT_DIR, "test", "test_data")
    if not os.path.isdir(save_dir_):
        os.makedirs(save_dir_)
    return save_dir_


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


def test_read_daymet_1basin_3days(save_dir):
    basin_id = "01013500"
    dates = ("2000-01-01", "2000-01-03")
    geometry = NLDI().get_basins(basin_id).geometry[0]
    # ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
    var = ["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"]
    daily = daymet.get_bygeom(geometry, dates, variables=var, pet=True)
    save_path = os.path.join(save_dir, basin_id + "_2000_01_01-03.nc")
    daily.to_netcdf(save_path)


def test_read_daymet_basins_3days(var, save_dir):
    basin_id = ["01013500", "01031500"]
    dates = ("2000-01-01", "2000-01-03")
    basins = NLDI().get_basins(basin_id)
    for i in range(len(basin_id)):
        # ``tmin``, ``tmax``, ``prcp``, ``srad``, ``vp``, ``swe``, ``dayl``
        daily = daymet.get_bygeom(basins.geometry[i], dates, variables=var)
        save_path = os.path.join(save_dir, basin_id[i] + "_2000_01_01-03.nc")
        daily.to_netcdf(save_path)


def test_download_nldi_shpfile(save_dir):
    basin_id = "01013500"
    basin = NLDI().get_basins(basin_id)
    # geometry = basin.geometry[0]
    save_path = os.path.join(save_dir, basin_id + ".shp")
    basin.to_file(save_path)

def test_equal_local_shp_download_shp_nc(save_dir):
    basin_id = "01013500"
    read_path = os.path.join(save_dir, basin_id + "_2000_01_01-03_nomask.nc")
    read_path_local = os.path.join(
        save_dir, basin_id + "_2000_01_01-03_nomask_local_shp.nc"
    )
    daily = xr.open_dataset(read_path)
    daily_local = xr.open_dataset(read_path_local)
    print(daily.equals(daily_local))

def test_batch_download_nldi_shpfile(save_dir):
    basin_id = ["01013500", "01031500"]
    basins = NLDI().get_basins(basin_id)
    # geometry = basin.geometry[0]
    save_path = os.path.join(save_dir, "two_test_basins.shp")
    basins.to_file(save_path)


