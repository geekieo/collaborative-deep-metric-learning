# Collaborative Deep Metric Learning for Video Understanding

## requirement

tensorflow-gpu    1.13.1  conda install tensorflow-gpu
faiss-gpu         1.5.1   conda install faiss-gpu cudatoolkit=9.0 -c pytorch  
  numpy           1.6.3

### optional

opencv-python     4.1
matplotlib
pillow
  
## Overview of Files

### Data

* `online_data.py`
* `parse_data.py`
* `imitation_data.py`
* ``

### Training

* `inputs.py`
* `models.py`
* `losses.py`
* `train.py`
* `utils.py`

### Evaluation

* `get_guid_title.py`
* `guid_knn.py`
* `average_precision_calculator.py`
* `mean_average_precision_calculator.py`

### Prediction

* `predict.py`

### Misc

* `README.md`
* `流程表.md`