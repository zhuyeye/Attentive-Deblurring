# Attentive deep network for blind motion deblurring in dynamic scenes
TensorFlow Implementation of CVIU paper "[Attentive deep network for blind motion deblurring in dynamic scenes](https://csyhquan.github.io/manuscript/21_cviu_Attentive%20Deep%20Network%20for%20Blind%20Motion%20Deblurring%20on%20Dynamic%20Scenes.pdf)" <br/>
## Dataset
We trained our model using the dataset from [DeepDeblur_release](https://github.com/SeungjunNah/DeepDeblur_release). Please put the training dataset into `training_set/`, and testing set into `testing_set/`.
## Test on our pretrain model
Our code is easy to go with:
```bash
python run_model.py --phase test --height 720 --width 1280 --gpu gpu_id
```
The quantitative results of **PSNR** and **SSIM** is calculted using MATLAB based on the deblurring results. Here we can get a **PSNR** result of about **31.22** with python codes.
## Training 
Training our model is easy to go with:
```bash
python run_model.py --phase train --batch batch_size --lr 0.0001 --epoch 4000
```

## Citation
If you think this work is useful for your research, please cite the following paper.

```
@article{XU2021103169,
title = {Attentive deep network for blind motion deblurring on dynamic scenes},
journal = {Computer Vision and Image Understanding},
volume = {205},
pages = {103169},
year = {2021},
issn = {1077-3142},
doi = {https://doi.org/10.1016/j.cviu.2021.103169},
url = {https://www.sciencedirect.com/science/article/pii/S1077314221000138},
author = {Yong Xu and Ye Zhu and Yuhui Quan and Hui Ji},
}
```

## Acknowledgement
Many parts of this code is adapted from [SRN](https://github.com/jiangsutx/SRN-Deblur)

Thanks the authors for sharing codes for their great works
