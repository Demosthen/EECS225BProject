# EECS225BProject

# Prerequisites

- Python 3.6, PyTorch >= 0.4
- Requirements: opencv-python, tqdm
- Platforms: Windows, [GPU/CPU NAME]
- To install the dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Run SelfDeblur

- SelfDeblur on Levin dataset. The code has been improved, and usually can achieve better retults than those reported in the paper.

```bash
python selfdeblur_levin.py
```

- SelfDeblur on Lai dataset, where blurry images have firstly been converted to their Y channel. Several images may converge to "black" deblurring images, but their estimated blur kernels are good. I will check why this happened. In these cases, you need to run `selfdeblur_nonblind.py` to generate final deblurring images.

```bash
python selfdeblur_lai.py
python selfdeblur_nonblind.py --data_path path_to_blurry --save_path path_to_estimated_kernel # Optional nonblind SelfDeblur. Given kernel estimated by Gk, only update Gx.
```

## Resources

- https://github.com/csdwren/SelfDeblur
- https://github.com/chrhenning/hypnettorch
