import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
import sys
import torch

#
need_monai=False
try:
    import monai
except ModuleNotFoundError:
    need_monai=False
if need_monai:
    os.system("pip install monai[all] -U")

#
need_pytorch_lightning=True
try:
    import pytorch_lightning
except ModuleNotFoundError:
    need_pytorch_lightning=True
if need_pytorch_lightning:
    os.system("pip install pytorch_lightning -U")

#
need_pytorch3d=True
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d=True
if need_pytorch3d:
    if torch.__version__.startswith("1.7") and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        version_str="".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".",""),
            f"_pyt{torch.__version__[0:5:2]}"
        ])
        os.system("pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html")
    else:
        # We try to install PyTorch3D from source.
        if not os.path.exists('cub-1.16.0'):
            os.system("wget https://github.com/NVIDIA/cub/archive/1.16.0.tar.gz")
            os.system("tar xzf 1.16.0.tar.gz")
        os.environ["CUB_HOME"] = "cub-1.16.0"
        os.system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git' -U --force-reinstall")


