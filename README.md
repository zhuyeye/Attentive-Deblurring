# Attentive-Deblurring
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
## Acknowledgement
Many parts of this code is adapted from [SRN](https://github.com/jiangsutx/SRN-Deblur)

Thans the authors for sharing codes for their great works
