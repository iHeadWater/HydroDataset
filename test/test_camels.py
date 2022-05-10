import os
import pytest

import numpy as np

import definitions
from hydrodataset.data.data_camels import Camels


@pytest.fixture
def camels_us_path():
    return os.path.join(definitions.DATASET_DIR, "camels", "camels_us")

@pytest.fixture
def us_region():
    return "US"



def test_download_camels(camels_us_path):
    camels_us = Camels(camels_us_path, download=True)
    assert os.path.isfile(
        os.path.join(camels_us_path, "basin_set_full_res", "HCDN_nhru_final_671.shp")
    )
    assert os.path.isdir(
        os.path.join(camels_us_path, "camels_streamflow", "camels_streamflow")
    )


def test_read_camels_streamflow(camels_us_path, us_region):
    camels_us = Camels(camels_us_path, download=False, region=us_region)
    gage_ids = camels_us.read_object_ids()
    flows1 = camels_us.read_target_cols(
        gage_ids[:5], ["2013-01-01", "2018-01-01"], target_cols=["usgsFlow"]
    )
    print(flows1)
    flows2 = camels_us.read_target_cols(
        gage_ids[:5], ["2015-01-01", "2018-01-01"], target_cols=["usgsFlow"]
    )
    print(flows2)


def test_read_camels_us(camels_us_path, us_region):
    camels_us = Camels(camels_us_path, download=False, region=us_region)
    gage_ids = camels_us.read_object_ids()
    assert gage_ids.size == 671
    attrs = camels_us.read_constant_cols(
        gage_ids[:5], var_lst=["soil_conductivity", "elev_mean", "geol_1st_class"]
    )
    np.testing.assert_almost_equal(
        attrs,
        np.array(
            [
                [1.10652248, 250.31, 10.0],
                [2.37500506, 92.68, 0.0],
                [1.28980735, 143.8, 10.0],
                [1.37329168, 247.8, 10.0],
                [2.61515428, 310.38, 7.0],
            ]
        ),
    )
    forcings = camels_us.read_relevant_cols(
        gage_ids[:5], ["1990-01-01", "2010-01-01"], var_lst=["dayl", "prcp", "srad"]
    )
    np.testing.assert_array_equal(forcings.shape, np.array([5, 7305, 3]))
    flows = camels_us.read_target_cols(
        gage_ids[:5], ["1990-01-01", "2010-01-01"], target_cols=["usgsFlow"]
    )
    np.testing.assert_array_equal(flows.shape, np.array([5, 7305, 1]))
    streamflow_types = camels_us.get_target_cols()
    np.testing.assert_array_equal(streamflow_types, np.array(["usgsFlow"]))
    focing_types = camels_us.get_relevant_cols()
    np.testing.assert_array_equal(
        focing_types, np.array(["dayl", "prcp", "srad", "swe", "tmax", "tmin", "vp"])
    )
    attr_types = camels_us.get_constant_cols()
    np.testing.assert_array_equal(
        attr_types[:3], np.array(["gauge_lat", "gauge_lon", "elev_mean"])
    )
