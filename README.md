# ACFormer
Code for Affine-Consistent Transformer for Multi-Class Cell Nuclei Detection (ICCV 2023)

(Continually updating ...)

## Overall Framework
![](./resources/framework.jpg)
## Requisities
-`python=3.8`

-`pytorch=1.12.1+cu102`


## Installation
Install mmcv using mim
```
pip install -U openmim
mim install mmcv-full==1.6.1
```
Git clone acformer
```
git clone https://github.com/LL3RD/ACFormer.git
```
Install
```
cd ACFormer
cd thirdparty/mmdetection 
python -m pip install -e .
cd ../.. 
python -m pip install -e .
```
## Dataset
### Lizard Dataset
Your can download [Original Lizard](https://warwick.ac.uk/fac/cross_fac/tia/data/lizard) from the official website or [Preprocessed Lizard](https://drive.google.com/file/d/1Rsr0rlKOHi7mqKBrmV3yOvXcF6g6BCY1/view?usp=sharing) that is converted to hovernet consep format and split into patches.


## Main Result
### Lizard Dataset
| Method | F1d | F1c | Model Weights |Config Files|
| ---- | -----| ----- |----|----|
| ACFormer | 0.782 | 0.557 | [Checkpoint](https://drive.google.com/file/d/12FyfAQf5VU2poXvqE_FmrB2HL6VDCldj/view?usp=sharing)|[Config](https://drive.google.com/file/d/14scJog5GjZc-n-Uwn4sIAJcaO2tokxA-/view?usp=sharing)|

### CoNSeP Dataset
| Method | F1d | F1c | Model Weights  |Config Files|
| ---- | -----| ----- |----------------|----|
| ACFormer | 0.739 | 0.613 | [Checkpoint](https://drive.google.com/file/d/1HHaVTvqVjh80mlQsBCdTEgqhRiCSHEIj/view?usp=sharing) |[Config](https://drive.google.com/file/d/1KyVHbeiSE4GOSFOE08d-XdeAB3-sftRr/view?usp=sharing)|

## Evaluation
Modify your dataset path and checkpoint path in tools/inference_lizard.py and run
```
python tools/inference_lizard.py
```

```
python tools/inference_consep.py
```

## Acknowledgement
- ACFormer is built based on [SoftTeacher](https://github.com/microsoft/SoftTeacher) and [MMDetection](https://github.com/open-mmlab/mmdetection).
