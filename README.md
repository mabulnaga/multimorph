# MultiMorph: On-demand Atlas Construction (CVPR 2025)
### [Project page](https://people.csail.mit.edu/abulnaga/multimorph/index.html) | [Paper](https://arxiv.org/abs/2504.00247) | [Colab notebook](https://colab.research.google.com/github/mabulnaga/multimorph/blob/main/src/build_3d_atlas.ipynb)

![Sample atlases constructed](https://people.csail.mit.edu/abulnaga/multimorph/teaser.png)

MultiMorph is an on-demand atlas construction neural network for 3D brain MRI. MultiMorph constructs atlases for a dataset in a single forward pass of the network, on the CPU, without requiring any re-training or fine tuning. MultiMorph is generally invariant to the modality as it was trained on T1w, T2w, and synthetic data covering a broad range of imaging contrasts.

This repository contains code to run MultiMorph on your data and construct an atlas in seconds to minutes. We also include demo CoLab notebookes showing how to train on 2D images from OASIS-1 and MNIST.

This repository contains:
- source code (in `./src/`) to train a MultiMorph model and code to construct an atlas (inference) using our pre-trained model.
- pre-trained model weights in `./models/`
- data in (`./data/`). This folder contains 2D slices from OASIS-1 (`./data/oasisdata/`) used in the tutorial. The folder also contains a few 3D volumes from OASIS-1 in `./data/oasis_3d_data/` to be used to test 3D atlas construction.


## Construct your Own Atlas Using Our Pre-trained Model 
We include a script to load our pre-trained model and run inference on your data. All you need is a CSV file with each row pointing to the image and (optionally) segmentation in your dataset.
To run,
```
python src/build_atlas_inference.py --model_path models/model_cvpr.pth --atlas_save_path path_to_save_atlas_files --csv_path path_to_csv_file --img_header_name header_name_img --segmentation_header_name -seg_header_name
```
The input arguments are:
- `model_path`: full file path to the model weights. The CVPR model is included in this repo under `models/model_cvpr.pth`
- `atlas_save_path`: full path to a folder on your system to save the output atlas and atlas segmentation files. By default, the code will save the outputs as `atlas_save_path/atlas.nii.gz` and `atlas_save_path/atlas_segmentation.nii.gz`
- `csv_path`: full file path to the CSV with your dataset information. The CSV file should have one or two columns. The first column will include full paths to each image file. The second (optional) column points to the corresponding segmentation files.
- `img_header_name`: name of the header in the CSV for the image files. By default, uses `img_path`
- `segmentation_header_name`: name of the header in the CSV for the segmentation files. If you do not have segmentations, pass in `None`, or don't include this argument. By default, it is set to `None`

We also include a tutorial CoLab notebook showing how to construct an atlas on 4 3D images from OASIS-1. However, we recommend running the code locally as the CoLab compute power is limited.

## Demo Tutorials
We include a tutorial notebooks for training your own MultiMorph model on 2D data.
- OASIS-1 2D training demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mabulnaga/multimorph/blob/main/src/demo_oasis1.ipynb)
We also include a tutorial notebook for running on inference on a pre-trained model to create a 3D atlas.
- OASIS-1 3D atlas construction (inference) demo: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mabulnaga/multimorph/blob/main/src/build_3d_atlas.ipynb)

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
