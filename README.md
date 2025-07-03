# USE-MiT
This repository contains the code of the method presented in the paper USE-MiT: Attention-based model for ultrasound images segmentation. The structure of the proposed model is based on a UNet architecture in which the encoder and decoder modules are interfaced through a configuration based on Squeeze and Excitation Attention modules (ED-SE blocks), and the encoder structure is represented by a Mix Transformer. ![USE-MiT](https://github.com/user-attachments/assets/74278e2b-7a30-4de3-9609-922039ee67b5)

# Experiments
The experiments presented in the paper USE-MiT: Attention-based model for ultrasound images segmentations are based on [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-datase) and [Breast-Lesions-USG (USG)](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/). For hyperparameter tuning, a 4-fold cross-validation protocol on the BUSI dataset was performed. Once the optimal hyperparameters were selected, the proposed model was trained on the BUSI dataset and evaluated on the independent USG dataset.

# Installation
The requirements.txt file should list all Python libraries that the present code depends on, and they will be installed using:

`pip install -r requirements.txt`

Once the `segmentation_models_pytorch` package is installed, the folder USE_MiT should be added in the directory `segmentation_models_pytorch/decoders`

# Running the code
General usage notes to run the script train_test.py containing the main are:

```
train_test.py  [--data_dir TRAINING IMAGES DIRECTORY] [--test_dir TESTING IMAGES DIRECTORY]
                  [--gpu_list GPU_LIST]
optional arguments:
--seed SEED                     Seed value               default 42
--learning_rate LEARNING_RATE   Learning rate value      default 0.0001
--num_epoch NUM_EPOCH           Number of epochs value   default 200
--batch BATCH SIZE              Batch size value         default 64
```

