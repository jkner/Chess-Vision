# **Chess-Vision**

The goal of this project is to convert a video of a live chess game to a pgn file.

## Demo

ScaledYOLOv4_tiny_detection_result:

## Installation

1. Clone Project
    
    ```jsx
    git clone https://github.com/jkner/Chess-Vision.git
    ```
    
2. Install Anaconda:
    
    ```jsx
    https://www.anaconda.com/
    ```

3. Install pip:

    ```jsx
    conda update --all
    conda install pip
    ```
4. Create an environment, activate it then install tensorflow-gpu:
    
    [Tensorflow Gpu :: Anaconda.org](https://anaconda.org/anaconda/tensorflow-gpu)
    

    ```jsx
    conda install tensorflow-gpu
    ```

5. Install requirements:

    ```jsx
    pip install -r requirements.txt
    ```

## Evaluation Results:

****Evaluation results(GTX3060Ti,mAP@0.5):****

| model                                             | Chess Pieces |
|--------------------------------------|----------------|   
| Scaled-YoloV4-tiny(Multi-Scale)    |        0.995      |        
| Scaled-YoloV4-tiny(Single-Scale)    |        0.989      | 
## Detection

To run the code:

```jsx
python3 cv_chess.py
```

## Detection Result
![Chess-Vision](https://github.com/jkner/Chess-Vision/blob/main/images/board_images/detected_board.png?raw=true)

### Video Demo
[![Chess-Vision](https://img.youtube.com/vi/Nw5VhdQbd-M/0.jpg)](https://www.youtube.com/watch?v=Nw5VhdQbd-M)

## References

### Board Detection & Piece Localization
Chess piece detection - digitalcommons.calpoly.edu. (n.d.). Retrieved May 14, 2022, from https://digitalcommons.calpoly.edu/cgi/viewcontent.cgi?article=1617&context=eesp

Underwood, A. (2020, October 22). Board Game Image Recognition Using Neural Networks. Medium. Retrieved May 13, 2022, from https://towardsdatascience.com/board-game-image-recognition-using-neural-networks-116fc876dafa

### Training and Validating Model:
wangermeng2021. (n.d.). Wangermeng2021/scaled-yolov4-tensorflow2: A tensorflow2.x implementation of scaled-yolov4 as described in scaled-yolov4: Scaling cross stage partial network. GitHub. Retrieved May 13, 2022, from https://github.com/wangermeng2021/Scaled-YOLOv4-tensorflow2 
