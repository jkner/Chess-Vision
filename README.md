# **# Chess-Vision**

```
[![Python 3.7](https://img.shields.io/badge/Python-3.7-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 2.4](https://img.shields.io/badge/TensorFlow-2.4-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
```

`A Tensorflow2.x implementation of Scaled-YOLOv4 as described in [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)`

## Demo

ScaledYOLOv4_tiny_detection_result:

## Installation

1. Clone Project
    
    ```jsx
    git clone https://github.com/jkner/Chess-Vision.git
    ```
    

1. Install Tensor flow:
    
    [Tensorflow Gpu :: Anaconda.org](https://anaconda.org/anaconda/tensorflow-gpu)
    

```jsx
conda install -c anaconda tensorflow-gpu
```

1. Install requirements:

```jsx
pip install -r requirements.txt
```

## Evaluation Results:

****Evaluation results(GTX2080,mAP@0.5):****

| model                                             | Chess Pieces |
|--------------------------------------|----------------|   
| Scaled-YoloV4-tiny(Multi-Scale)    |        0.995      |        

## Detection

To run the code:

```jsx
python3 cv_chess.py
```

## Detection Result