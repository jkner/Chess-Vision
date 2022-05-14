# **Chess-Vision**

The goal of this project is to convert a video of a live chess game to a pgn file.

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

****Evaluation results(GTX3060Ti,mAP@0.5):****

| model                                             | Chess Pieces |
|--------------------------------------|----------------|   
| Scaled-YoloV4-tiny(Multi-Scale)    |        0.995      |        

## Detection

To run the code:

```jsx
python3 cv_chess.py
```

## Detection Result
