# Attentive deep network for blind motion deblurring in dynamic scenes
TensorFlow Implementation of CVIU paper "[Attentive deep network for blind motion deblurring in dynamic scenes](https://csyhquan.github.io/manuscript/21_cviu_Attentive%20Deep%20Network%20for%20Blind%20Motion%20Deblurring%20on%20Dynamic%20Scenes.pdf)" <br/>
Refer to https://csyhquan.github.io/category/c_publication.html for our more publications.
## Dataset
We trained our model using the dataset from [DeepDeblur_release](https://github.com/SeungjunNah/DeepDeblur_release). Please put the training dataset into `training_set/`, and testing set into `testing_set/`.
## Test on our pretrain model
Our code is easy to go with:
```bash
python run_model.py --phase test --height 720 --width 1280 --gpu gpu_id
```
The quantitative results of **PSNR** and **SSIM** is calculted using MATLAB based on the deblurring results. Here we can get a **PSNR** result of about **31.22dB** with python codes.
## Training 
Training our model is easy to go with:
```bash
python run_model.py --phase train --batch batch_size --lr 0.0001 --epoch 4000
```

## Defocus Deblurring
Our model also works well on defocus deblurring, we train our model with [DPDD](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel) datatset and obtain a SOTA performance (about **25.66dB** on **PSNR**). The pretrained model is placed in `checkpoints/defocus/`. We only use the single color image as input, i.e. `train_c` and `test_c` of DPDD dataset.

To test our defocus deblurring performance on DPDD testing set, you can easy to go with:
```bash
python run_model.py --phase test --height 1120 --width 1680 --gpu gpu_id --model defcous --steps 105000 --input_path input_dir --output_path out_dir
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
author = {Yong Xu and Ye Zhu and Yuhui Quan and Hui Ji}
}
```

## Acknowledgement
Many parts of this code is adapted from [SRN](https://github.com/jiangsutx/SRN-Deblur)

Thanks the authors for sharing codes for their great works
