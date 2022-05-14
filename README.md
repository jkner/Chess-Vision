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
![alt text](https://github.com/jkner/Chess-Vision/images/board_images/detected_board.png)

## References

### Board Detection & Piece Localization
Chess piece detection - digitalcommons.calpoly.edu. (n.d.). Retrieved May 14, 2022, from https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=1617&context=eesp

Underwood, A. (2020, October 22). Board Game Image Recognition Using Neural Networks. Medium. Retrieved May 13, 2022, from https://towardsdatascience.com/board-game-image-recognition-using-neural-networks-116fc876dafa

### Training and Validating Model:
wangermeng2021. (n.d.). Wangermeng2021/scaled-yolov4-tensorflow2: A tensorflow2.x implementation of scaled-yolov4 as described in scaled-yolov4: Scaling cross stage partial network. GitHub. Retrieved May 13, 2022, from https://github.com/wangermeng2021/Scaled-YOLOv4-tensorflow2 
