<h1 align="center">
  Exploring Different Segmentation Models
</h1>

<p align="center">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"
         alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="pytorch"/>
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/>
  
</p>

<p align="center">
  <a href="#report">Report</a> •
  <a href="#model-performance">Model Performance</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a>
</p>

# Report
The report discusses about our methodology in implementing the segmentation models and evaluating it's performance based on their individual traits.
The report can be found in `report.pdf`.
# Model Performance
CLIP's powerful pre-trained encoder outperformed other models in segmenting images.
The segmentation models that will be used are:
| Model       | IoU  | Dice | Accuracy |
| :---------- | :--- | :--- | :------- |
| UNET        | 0.36 | 0.45 | 0.82     |
| Autoencoder | 0.23 | 0.30 | 0.77     |
| CLIP        | 0.55 | 0.61 | 0.94     |
| PointCLIP   | 0.37 | 0.53 | 0.56     |

# How to Use
The dataset can be found [here](https://www.robots.ox.ac.uk/~vgg/data/pets/).
More information about each folder can also be found in the report appendix.

## Setting up the Python environment
```py
# creating the environment
git clone https://github.com/yuhaopro/image-segmentation-cv.git

# installing dependencies
python3 -r requirements.txt

# adding project root to python
export PYTHONPATH="$PYTHONPATH:/path/to/projectroot"

# explore each model and run their test script
python3 clip/test.py
```

## Credits
This is part of my computer vision coursework with @Sylvia-678 in the University of Edinburgh. Hope you like it! 

