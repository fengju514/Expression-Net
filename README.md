# Expression-Net

This page contains a deep convolutional neural network (DCNN) model and python code for robust estimation of the 29 degrees-of-freedom, 3DMM face expression coefficients, directly from an unconstrained face image and without the use of face landmark detectors. The method is described in the paper: 

_F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[ExpNet: Landmark-Free, Deep, 3D Facial Expressions](https://arxiv.org/abs/1708.07517)", in the 13th IEEE Conference on Automatic Face and Gesture Recognition, 2018_ [1].

This release bundles up our **ExpressionNet** (ExpNet) with **FacePoseNet** (FPN) from Chang _et al._ [2], and **3DMM face identity shape network** from Tran _et al._ [3], which are available separately from the [FacePoseNet project page](https://github.com/fengju514/Face-Pose-Net) and the [Face Identity Shape Net project page](https://github.com/anhttran/3dmm_cnn), respectively. 

The code provided here bundels all three components for holistic 3D face modeling and produces a 3D model (.ply mesh file).

Mid level facial details can be added to this shape using our [extreme 3D reconstruction project](https://github.com/anhttran/extreme_3d_faces), but at the moment this is not supported by the code offered here. 

![Teaser](https://github.com/fengju514/Expression-Net/blob/master/ExpNet_teaser_v2.jpg)

## Features
* Estimating **29D 3DMM face expression coefficients**
* **3DMM face identity shape**[3] + **6DoF 3D head pose**[2] are also included ([facial details](https://github.com/anhttran/extreme_3d_faces) are a planned extenssion)
* Does not depend on fragile landmark detectors, therefore...
* ...robust under image conditions where landmark detectors struggle (low rez., occluded, etc.)
* Extremely fast expression estimation
* Provides better expression estmation than the ones using state-of-the-art landmark detectors [1]

## Dependencies

* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV Python Wrapper](http://opencv.org/)
* [Numpy](http://www.numpy.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)

The code has been tested on Linux with Python 2.7.12. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. 


## Usage

* **Important:** Please download the learned ExpressionNet and identity shape model from https://www.dropbox.com/s/ejrnujis19vxpmi/3DMM_model.tar.gz?dl=0, and FacePoseNet from https://www.dropbox.com/s/r38psbq55y2yj4f/fpn_new_model.tar.gz?dl=0. Make sure that the ExpNet and shape models are stored in `3DMM_model` and FPN models are stored in the folder `fpn_new_model`.

### Run it

To run it directly on a list of images (software will run ExpNet, ShapeNet, and PoseNet to estimate the expression, shape, and pose to get the .ply 3D mesh files). The final 3D shape can be displayed using standard off-the-shelf 3D (ply file) visualization software such as [MeshLab](http://meshlab.sourceforge.net)

```bash
$ python main_ExpShapePoseNet.py <input-list-path>
```

We provide a sample input list available [here](input.csv).
```bash
<ID, FILE, FACE_X, FACE_y, FACE_WIDTH, FACE_HEIGHT>
```
where `<FACE_X, FACE_y, FACE_WIDTH, FACE_HEIGHT>` is the face bounding box information, either obtained manually or by the face detector. 

## Sample Results
Please see the input images [here](images) and 3D shapes [here](output_ply).

![sbj10](https://github.com/fengju514/Expression-Net/blob/master/ExpNet_sample_results.jpg)



## Citation

This project is described in our paper [1]. If you use our expression models, please cite this paper using the bibtex below. If you also use the 3DMM face identity shape network [3] and FacePoseNet [2], pelase add references to those papers as well.

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
[1] F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[ExpNet: Landmark-Free, Deep, 3D Facial Expressions](https://arxiv.org/abs/1708.07517)", in the 13th IEEE Conference on Automatic Face and Gesture Recognition, 2018

[2] F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[FacePoseNet: Making a Case for Landmark-Free Face Alignment](https://arxiv.org/abs/1708.07517)", in the 7th IEEE International Workshop on Analysis and Modeling of Faces and Gestures, ICCV Workshops, 2017

[3] A. Tran, T. Hassner, I. Masi, G. Medioni, "[Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network](https://arxiv.org/abs/1612.04904)", in CVPR, 2017


## Changelog
- February 2018, First Release 

## Disclaimer

_The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use._

## Contacts

If you have any questions, drop an email to _fengjuch@usc.edu_, _anhttran@usc.edu_, _iacopoma@usc.edu_ or _hassner@isi.edu_ or leave a message below with GitHub (log-in is needed).
