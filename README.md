# MultiMorph: On-demand Atlas Construction (CVPR 2025)
### [Project page](https://people.csail.mit.edu/abulnaga/multimorph/index.html) | [Paper](https://arxiv.org/abs/2504.00247) | [Colab notebook](https://github.com/mabulnaga/multimorph/)

![Sample atlases constructed](https://people.csail.mit.edu/abulnaga/multimorph/teaser.png)

MultiMorph is an on-demand atlas construction neural network for 3D brain MRI. MultiMorph constructs atlases for a dataset in a single forward pass of the network, on the CPU, without requiring any re-training or fine tuning. MultiMorph is generally invariant to the modality as it was trained on T1w, T2w, and synthetic data covering a broad range of imaging contrasts.

This repository contains code to run MultiMorph on your data and construct an atlas in seconds to minutes. We also include demo CoLab notebookes showing how to train on 2D images from OASIS-1 and MNIST.

This repository contains:
- source code (in `./src/`) to train a MultiMorph model and code to construct an atlas (inference) using our pre-trained model.
- pre-trained model weights in `./models/`
- data in (`./data/`). This folder contains 2D slices from OASIS-1 (`./data/oasisdata/`) used in the tutorial. The folder also contains a few 3D volumes from OASIS-1 in `./data/oasis_3d_data/` to be used to test 3D atlas construction.


## Construct your Own Atlas Using Our Pre-trained Model 

## Demo Tutorials
We include two tutorial notebooks for training your own MultiMorph model on 2D data.
- OASIS-1 demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mabulnaga/multimorph/blob/main/src/demo_oasis1.ipynb)


## Training (Coming soon)

## Citation

If you find the paper or repository useful, please consider citing:

```
@inproceedings{abulnaga2025multimorph,
              title={MultiMorph: On-demand Atlas Construction},
              author={Abulnaga, S. Mazdak and Hoopes, Andrew and Dey, Neel and Hoffmann, Malte and Fischl, Bruce and Guttag, John and Dalca, Adrian},
              booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
              pages={30906--30917},
              year={2025}
              }
```
