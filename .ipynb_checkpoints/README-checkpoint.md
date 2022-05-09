# HydroDataset
下载并读取camels数据集
## 下载CAMELS数据集
下载camels_us数据集
```Python
import os
import definitions
from hydrodataset.data.data_camels import Camels

# DATASET_DIR is defined in the definitions.py file
camels_path = os.path.join(definitions.DATASET_DIR, "camels", "camels_us")
camels = Camels(camels_path, download=True)
```
## 运行code
```Shell
conda env create -f environment.yml
conda activate HydroDataset
```
