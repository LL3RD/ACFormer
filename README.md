# ACFormer
Code for Affine-Consistent Transformer for Multi-Class Cell Nuclei Detection (ICCV 2023)

## Requisities
-`python=3.8`

-`pytorch=1.12.1+cu102`


## Installation
Install mmcv using mim
```
pip install -U openmim
mim install mmcv==1.6.1
```
Git clone acformer
```
git clone 
```
Install
```
cd ACFormer
cd thirdparty/mmdetection 
python -m pip install -e .
cd ../.. 
python -m pip install -e .
```




## Acknowledgement
- ACFormer is built based on [SoftTeacher](https://github.com/microsoft/SoftTeacher) and [MMDetection](https://github.com/open-mmlab/mmdetection).
