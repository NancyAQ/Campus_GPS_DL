# Campus GPS : Coordinates regression task from an RGB image

This project explores GPS coordinate regression from campus images using deep learning.
It provides tools for dataset annotation, data splitting, model training (Swin-T), evalution and inference and visualization.

All commands should be run from the project root directory.
Our best checkpoint is saved in the CheckPoints folder under: swin_t_CampGPS_best.pt
## its very important for the dataset folder to be of the folliwng hierarchy 
## Dataset Structure

dataset/
├── images/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── gt.csv
st the splits files have the names of the images from the images folder.
## Make sure you have a CheckPoints folder in the root of the project to save the check points in, you can download our swin checkpoint from here
(https://drive.google.com/drive/folders/1Z1qthscxODZeG0w2fMDuvKz-0UiU1k8e?usp=sharing)
## 1. Environment Setup

Create and activate a conda environment before running any scripts:

    conda create -n campus_gps python=3.10
    conda activate campus_gps
    pip install -r requirements.txt

All training and inference must be executed inside this environment to support modules we use.



## 2. Dataset annotation or extension

If you want to create your own dataset or add new images to the existing one, run the
preprocessing script.

This script:
- Extracts EXIF metadata( make sure to replace the exiftiool path to yours)
- Creates a gt CSV file with GPS coordinates: gt.csv

SCRIPT : python -m src.data_procesing.data_preprocess


## 3. Dataset split to train, validate and test:

To split the dataset:

- Make sure there is a folder named splits/ inside the dataset/ directory

Run:

    python -m src.data_procesing.split_data

This will generate 3 splits with 60,20 and 20 ratio:
- train.txt
- val.txt
- test.txt



## 4. Model training- to train our best performong model the Swin-T
First make sure you have a folder names CheckPoints in your project folder
Our best performing model is Swin-T, trained using:

    src/training/SWIN_train.py

Before training, make sure:
- The dataset exists
- gt.csv has been generated
- The dataset has already been split 

Run training:

    python -m src.training.SWIN_train

### cluster usage/computing nodes usage for run and forget method

If working on a cluster, an example SBATCH file is provided:

    swin.sbatch 
make sure you edit it according to your user name and correct paths, you can do so by calling nano swin.sbatch

To manually allocate a GPU session for interactivve work, this is only recommended for quick inference testing(1 hour):

    srun --partition=main --time=1:00:00 --gpus=1 --pty bash -i


## 5. Evaluation on Test Set

After training is complete, evaluate the model on the test set:

    python -m src.evaluation.eval

This reports mean and median localization error and also save the results in outputs/Swin_T/test_res.csv
We have the best checkpoint path statically defined in the glocal variables alongside the dataset and the split file, if you wish to change those make sure you also change these.


## 6.Prediction

To predict GPS coordinates for a single image, use:

    src/inference/predict_gps.py

Run the script with the image path as the first argument:

    python -m src.inference.predict_gps <path_to_image>

Example:

    python -m src.inference.predict_gps dataset/images/IMG_5785.jpg

The script outputs the predicted latitude and longitude.
## 7. Plotting the cdf of the model results
 we provide a code in src/vis/plot.py to plot the cdf of the results from running the model on the test set
 to use this you will need to run step 5 and have outputs/Swin_T/test_res.csv in your project folder.
 SCRIPT: python -m src.vis.plot

 ## 8 visualizing results with opencv(to be saved since cluster does not support active visualiation)

We visualize model predictions by overlaying the predicted GPS coordinates, ggt coordinates, and localization error in meters for each of the test images. 
Before running this you will need these:

1. Dataset annotated and in the format described above with a gt file.
2. Dataset split.
3. Model trained (Swin-T)
4. Evaluation: eval model on test set with this existing outputs/Swin_T/test_res.csv
SCRIPT: python -m src.vis.vis






