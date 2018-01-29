# Expression-Net

![Teaser](http://www-bcf.usc.edu/~iacopoma/img/FPN4.jpg)
<sub>**Extreme face alignment examples:** Faces rendered to a 45 degrees yaw angle (aligned to half profile) using our FacePoseNet. Images were taken from the IJB-A collection and represent extreme viewing conditions, including near profile views, occlusions, and low resolution. Such conditions are often too hard for existing face landmark detection methods to handle yet easily aligned with our FacePoseNet.</sub>
<br/>
<br/>
This page contains DCNN model and python code to robustly estimate 6 degrees of freedom, 3D face pose from an unconstrained image, without the use of face landmark detectors. The method is described in the paper:

_F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[FacePoseNet: Making a Case for Landmark-Free Face Alignment](https://arxiv.org/abs/1708.07517)", in 7th IEEE International Workshop on Analysis and Modeling of Faces and Gestures, ICCV Workshops, 2017_ [1].

This release bundles up our **FacePoseNet** (FPN) with the **Face Renderer** from Masi _et al._ [2,5], which is available separately from [this project page](https://github.com/iacopomasi/face_specific_augm).

The result is an end-to-end pipeline that seamlessly estimates facial pose and produces multiple rendered views to be used for face alignment and data augmentation.

![Teaser](http://www-bcf.usc.edu/~iacopoma/img/FPN3.png)

## Features
* **6DoF 3D Head Pose estimation** + **3D rendered facial views**.
* Does not use **fragile** landmark detectors
* Robustness on images landmark detectors struggle with (low rez., occluded, etc.)
* Extremely fast pose estimation
* Both CPU and GPU supported
* Provides better face recognition through better face alignment than alignment using state of the art landmark detectors [1]

## Dependencies

* [TensorFlow](https://www.tensorflow.org/)
* [OpenCV Python Wrapper](http://opencv.org/)
* [Numpy](http://www.numpy.org/)
* [Python2.7](https://www.python.org/download/releases/2.7/)

The code has been tested on Linux only. On Linux you can rely on the default version of python, installing all the packages needed from the package manager or on Anaconda Python and install required packages through `conda`. 

**Note:** no landmarks are used in our method, although you can still project the landmarks on the input image using the estimated pose. See the paper for further details. 

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
FPN is currently trained with a single 3D generic shape, without accounting for facial expressions. Addressing these is planned as future work.

## Citation

Please cite our paper with the following bibtex if you use our face renderer:

``` latex
@inproceedings{chang17fpn,
      title={{F}ace{P}ose{N}et: Making a Case for Landmark-Free Face Alignment},
      booktitle = {7th IEEE International Workshop on Analysis and Modeling of Faces and Gestures, ICCV Workshops},
      author={
      Feng-ju Chang
      and Anh Tran 
      and Tal Hassner 
      and Iacopo Masi 
      and Ram Nevatia
      and G\'{e}rard Medioni},
      year={2017},
    }
```

## References
[1] F.-J. Chang, A. Tran, T. Hassner, I. Masi, R. Nevatia, G. Medioni, "[FacePoseNet: Making a Case for Landmark-Free Face Alignment](https://arxiv.org/abs/1708.07517)", in 7th IEEE International Workshop on Analysis and Modeling of Faces and Gestures, ICCV Workshops, 2017

[2] I. Masi\*, A. Tran\*, T. Hassner\*, J. Leksut, G. Medioni, "Do We Really Need to Collect Million of Faces for Effective Face Recognition? ", ECCV 2016, 
    \* denotes equal authorship

[3] I. Masi, S. Rawls, G. Medioni, P. Natarajan "Pose-Aware Face Recognition in the Wild", CVPR 2016

[4] T. Hassner, S. Harel, E. Paz and R. Enbar "Effective Face Frontalization in Unconstrained Images", CVPR 2015

[5] I. Masi, T. Hassner, A. Tran, and G. Medioni, "Rapid Synthesis of Massive Face Sets for Improved Face Recognition", FG 2017

## Changelog
- August 2017, First Release 

## Disclaimer

_The SOFTWARE PACKAGE provided in this page is provided "as is", without any guarantee made as to its suitability or fitness for any particular use. It may contain bugs, so use of this tool is at your own risk. We take no responsibility for any damage of any sort that may unintentionally be caused through its use._

## Contacts

If you have any questions, drop an email to _fengjuch@usc.edu_, _anhttran@usc.edu_, _iacopo.masi@usc.edu_ or _hassner@isi.edu_ or leave a message below with GitHub (log-in is needed).
