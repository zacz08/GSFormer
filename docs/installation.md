# Installation

Tested environment: Python 3.12, CUDA 12.8, PyTorch 2.7.1.

## 1. Create conda environment
```bash
conda create -n gsformer python=3.12 -y
conda activate gsformer
```

## 2. Install PyTorch
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

## 3. Upgrade setuptools
Python 3.12 removed `pkgutil.ImpImporter`, so a modern setuptools is required:
```bash
pip install --upgrade setuptools pip wheel
```

## 4. Install packages from MMLab
> **Note:** mmcv 2.2.0 will be compiled from source (takes a few minutes).
> mmseg/mmdet/mmdet3d have hard-coded version upper bounds that are too low;
> after installing them, we patch the version checks (see step below).

```bash
pip install mmengine==0.10.7
pip install mmcv==2.2.0
pip install mmsegmentation==1.2.2 mmdet==3.3.0
pip install shapely                        # pre-install to avoid build conflict
pip install nuscenes-devkit                 # pre-install to avoid build conflict
pip install mmdet3d==1.4.0 --no-deps       # skip deps to avoid old Shapely source build
```

### Patch MMLab version constraints
mmseg/mmdet/mmdet3d reject mmcv>=2.2.0 and mmengine>=1.0.0 at import time.
Run the following **once** to relax the upper bounds:
```bash
SP=$(python -c "import site; print(site.getsitepackages()[0])")

# mmseg
sed -i "s/MMCV_MAX = '2.2.0'/MMCV_MAX = '2.3.0'/" "$SP/mmseg/__init__.py"
sed -i "s/MMENGINE_MAX = '1.0.0'/MMENGINE_MAX = '1.1.0'/" "$SP/mmseg/__init__.py"

# mmdet
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$SP/mmdet/__init__.py"
sed -i "s/mmengine_maximum_version = '1.0.0'/mmengine_maximum_version = '1.1.0'/" "$SP/mmdet/__init__.py"

# mmdet3d
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$SP/mmdet3d/__init__.py"
sed -i "s/mmengine_maximum_version = '1.0.0'/mmengine_maximum_version = '1.1.0'/" "$SP/mmdet3d/__init__.py"
```

## 5. Install other packages
```bash
pip install spconv-cu126    # cu128 wheel not available; cu126 is forward-compatible
pip install timm einops ftfy regex jaxtyping tensorboard
```

## 6. Install custom CUDA ops
Use `--no-build-isolation` so that the build can find PyTorch in the current env:
```bash
cd model/encoder/gaussian_encoder/ops && pip install -e . --no-build-isolation && cd -
cd model/head/localagg && pip install -e . --no-build-isolation && cd -
# for GaussianFormer-2
cd model/head/localagg_prob && pip install -e . --no-build-isolation && cd -
cd model/head/localagg_prob_fast && pip install -e . --no-build-isolation && cd -
```

## 7. (Optional) For visualization
```bash
pip install pyvirtualdisplay mayavi matplotlib PyQt5
```