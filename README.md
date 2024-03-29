# A heterogenous group CNN for image super-resolution (HGSRCNN)
## This paper is conducted by Chunwei Tian (IEEE Member), Yanning Zhang (IEEE Senior Member), Wangmeng Zuo (IEEE Senior Member), Chia-Wen Lin (IEEE Fellow), David Zhang (IEEE Life Fellow) and Yixuan Yuan (IEEE Member). This paper is accepted by the IEEE Transactions on Neural Networks and Learning Systems (SCI-IF:14.255).  This paper can be obtained at https://arxiv.org/abs/2209.12406. It is reported by Extreme Mart at https://mp.weixin.qq.com/s/LJqwsATVijxkDhNdCTt42Q and AIWalker at https://mp.weixin.qq.com/s/3zjTZuHF2uJ-ihi07O8b5g. 

## It is invited to conduct a benckmark of super-resolution. 

## Absract
#### Convolutional neural networks (CNNs) have obtained remarkable performance via deep architectures. However, these CNNs often achieve poor robustness for image super-resolution (SR) under complex scenes. In this paper, we present a heterogeneous group SR CNN (HGSRCNN) via leveraging structure information of different types to obtain a high-quality image. Specifically, each heterogeneous group block (HGB) of HGSRCNN uses a heterogeneous architecture containing a symmetric group convolutional block and a complementary convolutional block in a parallel way to enhance internal and external relations of different channels for facilitating richer low-frequency structure information of different types. To prevent appearance of obtained redundant features, a refinement block with signal enhancements in a serial way is designed to filter useless information. To prevent loss of original information, a multi-level enhancement mechanism guides a CNN to achieve a symmetric architecture for promoting expressive ability of HGSRCNN. Besides, a parallel up-sampling mechanism is developed to train a blind SR model. Extensive experiments illustrate that the proposed HGSRCNN has obtained excellent SR performance in terms of both quantitative and qualitative analysis. Codes can be accessed at https://github.com/hellloxiaotian/HGSRCNN.


https://user-images.githubusercontent.com/25679314/192536076-f5122657-5f8e-4d39-9490-1aee0f94ce9d.mp4



## Requirements (Pytorch)  
#### Pytorch 0.41

#### Python 2.7

#### torchvision

#### torchsummary

#### openCv for Python

#### HDF5 for Python

#### Numpy, Scipy

#### Pillow, Scikit-image

#### importlib

## Commands
### Training datasets

#### The training dataset is downloaded at https://pan.baidu.com/s/1uqdUsVjnwM_6chh3n46CqQ （secret code：auh1）(baiduyun) or https://drive.google.com/file/d/1TNZeV0pkdPlYOJP1TdWvu5uEroH-EmP8/view (google drive)

### Test datasets

#### The test dataset of Set5 is downloaded at 链接：https://pan.baidu.com/s/1YqoDHEb-03f-AhPIpEHDPQ (secret code：atwu) (baiduyun) or https://drive.google.com/file/d/1hlwSX0KSbj-V841eESlttoe9Ew7r-Iih/view?usp=sharing (google drive)

#### The test dataset of Set14 is downloaded at 链接：https://pan.baidu.com/s/1GnGD9elL0pxakS6XJmj4tA (secret code：vsks) (baiduyun) or https://drive.google.com/file/d/1us_0sLBFxFZe92wzIN-r79QZ9LINrxPf/view?usp=sharing (google drive)

#### The test dataset of B100 is downloaded at 链接：https://pan.baidu.com/s/1GV99jmj2wrEEAQFHSi8jWw （secret code：fhs2) (baiduyun) or https://drive.google.com/file/d/1G8FCPxPEVzaBcZ6B-w-7Mk8re2WwUZKl/view?usp=sharing (google drive)

#### The test dataset of Urban100 is downloaded at 链接：https://pan.baidu.com/s/15k55SkO6H6A7zHofgHk9fw (secret code：2hny) (baiduyun) or https://drive.google.com/file/d/1yArL2Wh79Hy2i7_YZ8y5mcdAkFTK5HOU/view?usp=sharing (google drive)

### preprocessing

### cd dataset

### python div2h5.py

### Training a model for different scales (also regarded as blind SR)

#### python train.py --patch_size 83 --batch_size 32 --max_steps 600000 --decay 400000 --model HGSRCNN --ckpt_name HGSRCNN --ckpt_dir checkpoint/HGSRCNN --scale 0 --num_gpu 1

### Using a model to test different scales of 2,3 and 4 (also regarded as blind SR)

#### python tcw_sample.py --model HGSRCNN --test_data_dir dataset/Set5 --scale 2 --ckpt_path checkpoint/HGSRCNN.pth --sample_dir samples_singlemodel_urban100_x2

#### python tcw_sample.py --model HGSRCNN --test_data_dir dataset/Set5 --scale 3 --ckpt_path checkpoint/HGSRCNN.pth --sample_dir samples_singlemodel_urban100_x3

#### python tcw_sample.py --model HGSRCNN --test_data_dir dataset/Set5 --scale 4 --ckpt_path checkpoint/HGSRCNN.pth --sample_dir samples_singlemodel_urban100_x4

## 1. Network architecture of HGSRCNN

![Network architecture of HGSRCNN](./img/Network architecture of HGSRCNN.png)

## 2. Architecture of a parallel up-sampling mechanism

<img src="./img/Architecture of a parallel up-sampling mechanism.png" alt="Architecture of a parallel up-sampling mechanism" style="zoom:50%;" />

## 3. HGSRCNN for x2，x3 and x4 on Set5

<img src="./img/Set5.png" alt="Set5" style="zoom:67%;" />

## 4. HGSRCNN for x2，x3 and x4 on Set14

<img src="./img/Set14.png" alt="Set14" style="zoom:67%;" />

## 5. HGSRCNN for x2，x3 and x4  on B100

<img src="./img/B100.png" alt="B100" style="zoom:67%;" />

## 6. HGSRCNN for x2，x3 and x4  on U100

<img src="./img/U100.png" alt="U100" style="zoom:67%;" />

## 7. Running time of different methods on hr images of size 256x256, 512x512 and 1024x1024 for x2.

![Running time](./img/Running time.png)

## 8. Complexities of different methods for x2.

![Complexity](./img/Complexity.png)

## 9. ESRGCNN for x2, x3 and x4 on B100 about FSIM.

![FSIM](./img/FSIM.png)

## 10. Visual results of U100 for x3.

<img src="./img/VU100.png" alt="VU100" style="zoom:67%;" />

## 11. Visual results of B100 for x4.

![VB00](./img/VB00.png)

## You can cite this paper, according to the following information. 
## 1. Tian C, Zhang Y, Zuo W, et al. A heterogeneous group CNN for image super-resolution[J]. arXiv preprint arXiv:2209.12406, 2022.
## 2. @article{tian2022heterogeneous,
##   title={A heterogeneous group CNN for image super-resolution},
##   author={Tian, Chunwei and Zhang, Yanning and Zuo, Wangmeng and Lin, Chia-Wen and Zhang, David and Yuan, Yixuan},
##   journal={arXiv preprint arXiv:2209.12406},
##   year={2022}
##   }


