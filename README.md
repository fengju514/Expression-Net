# Expression-Net

This page contains a deep convolutional neural network (DCNN) model and python code for robust estimation of the 29 degrees-of-freedom, 3DMM face expression coefficients, directly from an unconstrained face image and without the use of face landmark detectors. The method is described in the paper: 

_F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[ExpNet: Landmark-Free, Deep, 3D Facial Expressions](https://arxiv.org/abs/1802.00542)", in the 13th IEEE Conference on Automatic Face and Gesture Recognition, 2018_ [1].

This release bundles up our **ExpressionNet** (ExpNet) with **FacePoseNet** (FPN) from Chang _et al._ [2], and **3DMM face identity shape network** from Tran _et al._ [3], which are available separately from the [FacePoseNet project page](https://github.com/fengju514/Face-Pose-Net) and the [Face Identity Shape Net project page](https://github.com/anhttran/3dmm_cnn), respectively. 

The code provided here bundles all three components for holistic 3D face modeling and produces a 3D model (.ply mesh file).

**Important** This is an ongoing project. Please check here for updates, corrections and extensions. In particular, mid level facial details can be added to this shape using our [extreme 3D reconstruction project](https://github.com/anhttran/extreme_3d_faces). At the moment facial details estimation is not supported by the code offered here, but we plan to add this in the future. 

![Teaser](https://github.com/fengju514/Expression-Net/blob/master/ExpNet_teaser_v2.jpg)

## Features
* Estimating **29D 3DMM face expression coefficients**
* **3DMM face identity shape** [3] + **6DoF 3D head pose** [2] are also included ([facial details estimation](https://github.com/anhttran/extreme_3d_faces) is a planned extension)
* Does not depend on fragile landmark detectors, therefore...
* ...robust under image conditions where landmark detectors struggle such as low resolutions and occlusions
* Extremely fast expression estimation
* Provides better expression estimation than the ones using state-of-the-art landmark detectors [1]

## Dependencies

* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV Python Wrapper](http://opencv.org/)
* [Numpy](http://www.numpy.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)

The code has been tested on Linux with Python 2.7.12. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. 


## Usage

**Important:** Please download the following learned deep models:
* ExpressionNet from https://www.dropbox.com/s/frq7u7z5kgxnz9e/Expression_Model.tar.gz?dl=0
* Identity shape model from https://www.dropbox.com/s/ej80o9lnj0k49qu/Shape_Model.tar.gz?dl=0, and
* FacePoseNet from https://www.dropbox.com/s/r38psbq55y2yj4f/fpn_new_model.tar.gz?dl=0. 
Make sure that the ExpNet, shape, and FacePoseNet models are stored in the folder `Expression_Model`, `Shape_Model`, and `fpn_new_model` respectively.

### Run it

Our code use a list of images as an input. The software will run ExpNet, ShapeNet, and PoseNet to estimate the expression, shape, and pose to get the .ply 3D mesh files for the images in this list. The final 3D shape can be displayed using standard off-the-shelf 3D (ply file) visualization software such as [MeshLab](http://meshlab.sourceforge.net)

```bash
$ python main_ExpShapePoseNet.py <input-list-path>
```

We provide a sample input list available [here](input.csv). More specifically, it is a text file with multiple lines. Each line provides information of an input image with this format:

```bash
<ID, FILE, FACE_X, FACE_y, FACE_WIDTH, FACE_HEIGHT>
```
where `<FACE_X, FACE_y, FACE_WIDTH, FACE_HEIGHT>` is the face bounding box information, either obtained manually or by the face detector. 

## Sample Results
Please see the input images ([images](images)), cropped images ([tmp](tmp)), and the output 3D shapes ([output_ply](output_ply)). 

![sample_res](https://github.com/fengju514/Expression-Net/blob/master/ExpNet_sample_results.jpg)
Note: Rendered 3D results appearing here were not rendered to the same scale as the face in the input views, though the pose estimation does provide this information.

## Citation

This project is described in our paper [1]. If you use our expression model, please cite this paper using the bibtex below. If you also use the 3DMM face identity shape network [3] and FacePoseNet [2], please add references to those papers as well.

``` latex
@inproceedings{chang17expnet,
      title={ExpNet: Landmark-Free, Deep, 3D Facial Expressions},
      booktitle = {13th IEEE Conference on Automatic Face and Gesture Recognition},
      author={
      Feng-Ju Chang
      and Anh Tran 
      and Tal Hassner 
      and Iacopo Masi 
      and Ram Nevatia
      and G\'{e}rard Medioni},
      year={2018},
    }
```

## References
[1] F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[ExpNet: Landmark-Free, Deep, 3D Facial Expressions](https://arxiv.org/abs/1802.00542)", in the 13th IEEE Conference on Automatic Face and Gesture Recognition, 2018

[2] F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[FacePoseNet: Making a Case for Landmark-Free Face Alignment](https://arxiv.org/abs/1708.07517)", in the 7th IEEE International Workshop on Analysis and Modeling of Faces and Gestures, ICCV Workshops, 2017

[3] A. Tran, T. Hassner, I. Masi, G. Medioni, "[Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network](https://arxiv.org/abs/1612.04904)", in CVPR, 2017


## Changelog
- February 2018, First Release 

## Disclaimer

_The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use._

## Contacts

If you have any questions, drop an email to _fengjuch@usc.edu_, _anhttran@usc.edu_, _iacopoma@usc.edu_ or _hassner@isi.edu_ or leave a message below with GitHub (log-in is needed).
