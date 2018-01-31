# EXPression-Net

This page contains DCNN model and python code to robustly estimate 29 degrees of freedom, 3DMM face expression parameter from an unconstrained image, without the use of face landmark detectors. The method is described in the paper:

_F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[ExpNet: Landmark-Free, Deep, 3D Facial Expressions](https://arxiv.org/abs/1708.07517)", in the 13th IEEE Conference on Automatic Face and Gesture Recognition, 2018_ [1].

This release bundles up our **ExpressionNet** (ExpNet) with **FacePoseNet** (FPN) from Chang_et al._ [2], and the state-of-the-art **3DMM face identity (shape) parameter model** from Tran _et al._ [3], which are available in [this project page](https://github.com/fengju514/Face-Pose-Net) and [this project page](https://github.com/anhttran/3dmm_cnn) respectively.

The result is an end-to-end pipeline that seamlessly estimates facial expression, pose and shape for holistic 3D face modeling.

![Teaser](http://www-bcf.usc.edu/~iacopoma/img/FPN3.png)

## Features
* **29DoF 3DMM face expression estimation** + **6DoF 3D head pose estimation**.
* Does not use **fragile** landmark detectors
* Robustness on images landmark detectors struggle with (low rez., occluded, etc.)
* Extremely fast expression and pose estimation
* Both CPU and GPU supported
* Provides better expression estmation than the ones using state of the art landmark detectors [1]

## Dependencies

* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV Python Wrapper](http://opencv.org/)
* [Numpy](http://www.numpy.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)

The code has been tested on Linux only. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. 

**Note:** no landmarks are used in our method, although you can still project the landmarks on the input image using the estimated expression and pose. See the paper for further details. 

## Usage

* **Important:** In order to download **both** FPN code and the renderer use `git clone --recursive`
* **Important:** Please download the learned models from https://www.dropbox.com/s/r38psbq55y2yj4f/fpn_new_model.tar.gz?dl=0   and make sure that the FPN models are stored in the folder `fpn_new_model`.

### Run it

The alignment and rendering can be used from the command line in the following, different ways.

To run it directly on a list of images (software will run FPN to estimate the pose and then render novel views based on the estimated pose):

```bash
$ python main_fpn.py <input-list-path>
```

We provide a sample input list available [here](input.csv).
```bash
<ID, FILE, FACE_X, FACE_y, FACE_WIDTH, FACE_HEIGHT>
```
where `<FACE_X, FACE_y, FACE_WIDTH, FACE_HEIGHT>` is the face bounding box information, either obtained manually or by the face detector. 

## Sample Results
Please see the input images [here](images) and rendered outputs [here](output_render).

### input: ### 
![sbj10](./images/input10.jpg)
### rendering: ### 
![sbj10](./output_render/subject10/subject10_a_rendered_aug_-00_00_10.jpg)
![sbj10](./output_render/subject10/subject10_a_rendered_aug_-22_00_10.jpg)
![sbj10](./output_render/subject10/subject10_a_rendered_aug_-40_00_10.jpg)
![sbj10](./output_render/subject10/subject10_a_rendered_aug_-55_00_10.jpg)
![sbj10](./output_render/subject10/subject10_a_rendered_aug_-75_00_10.jpg)



## Current Limitations

## Citation

Please cite our paper with the following bibtex if you use our code:

``` latex
@inproceedings{chang17fpn,
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

If you have any questions, drop an email to _fengjuch@usc.edu_, _anhttran@usc.edu_, _iacopo.masi@usc.edu_ or _hassner@isi.edu_ or leave a message below with GitHub (log-in is needed).
