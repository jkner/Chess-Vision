import re
import cv2
import numpy as np
from cv_chess_functions import (read_img,
                                canny_edge,
                                hough_line,
                                h_v_lines,
                                line_intersections,
                                cluster_points,
                                augment_points,
                                fen_to_image,
                                crop_image,
                                remove_duplicates,
                                remove_outside_points,
                                atoi,
                                trans_boxes)

from detect import (main,
                    tensor)
import chess


# Resize the frame by scale by dimensions
def rescale_frame(frame, percent=75):
    # width = int(frame.shape[1] * (percent / 100))
    # height = int(frame.shape[0] * (percent / 100))
    dim = (416, 416)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# Finding Corners
def detect_corners():
    # Low-level CV techniques (grayscale & blur)
    img, gray_blur = read_img('./images/chess_pictures/cropped_frame.jpeg')
    #   img, gray_blur = read_img('detect.jpeg')
    # Canny algorithm
    edges = canny_edge(gray_blur)
    # Hough Transform
    lines = hough_line(edges)
    # Separate the lines into vertical and horizontal lines
    h_lines, v_lines = h_v_lines(lines)
    # Find and cluster the intersecting
    intersection_points = line_intersections(h_lines, v_lines)

    # two ways to cluster
    cluster1 = cluster_points(intersection_points)
    cluster2 = remove_duplicates(intersection_points)

    # Final coordinates of the board
    aug_points = np.array(augment_points(cluster2))

    # Remove Outside Points:
    inner_points1 = remove_outside_points(cluster1)
    inner_points2 = remove_outside_points(aug_points)

    # Draws Chessboard Corners using corner points
    drawn = cv2.drawChessboardCorners(img, (9, 9), inner_points2, True)

    # cv2.imwrite('./corner1.jpeg', drawn)
    cv2.imshow('Corners', drawn)

    num_of_corners = len(inner_points2)
    print("num of corners", num_of_corners)
    return drawn, inner_points2


def save_crop_img():
    ret, frame = cap.read()
    cv2.imshow('live', frame / 255)
    out = cv2.addWeighted(frame, 1, frame, 0, 1)
    cv2.imwrite('frame.jpeg', out)
    crop_frame = crop_image(out)
    cv2.imwrite('./images/chess_pictures/cropped_frame.jpeg', crop_frame)


# Calibrate board
def calibrate_board(calibrated):
    while not calibrated:
        print('Calibrating Board....')
        save_crop_img()
        if cv2.waitKey(0) & 0xFF == ord('c'):
            img, corner_points = detect_corners()
            if cv2.waitKey(0) & 0xFF == ord('s'):
                print("saving corner image...")
                cv2.imwrite('corner.jpeg', img)
                save_corner_points(corner_points)
                return


def save_corner_points(corner_points):
    rows, cols = (9, 9)
    arr = [[0] * cols] * rows

    array_int = np.array(corner_points).astype(int)
    #print("array int", array_int)
    #
    # print("corner points", corner_points)
    # print("index 0", array_int[0]) # index 0 of corner points
    # print("index 0, 0?", array_int[0][0]) # index 0 of corner points
    #
    # arr[0][0] = array_int[0]
    # print((arr[0][0])[0])
    # np.array(arr)
    #
    # for index in array_int:
    # arr[0][0] = array_int[0]
    # arr[0][1] = array_int[1]
    # arr[0][2] = array_int[2]
    # arr[0][3] = array_int[3]
    # arr[0][4] = array_int[4]
    # arr[0][5] = array_int[5]
    # arr[0][6] = array_int[6]
    # arr[0][7] = array_int[7]
    # arr[0][8] = array_int[8]
    # arr[1][0] = array_int[9]
    i = 0
    for j in range(0, 9):
        for z in range(81):
            arr[i][j] = array_int[z]
            print(arr[i][j])


    # accessing index 0 of arr[0][0]
    # arr[0][1] = [121.22601   44.00001 ] = corner_points[1]
    # arr[0][2] = [168.92873   44.00001 ] = corner_points[]
    # for index in corner_points:
    #     for i in arr:
    #         for j in arr:
    #             arr[i][j] = corner_points[index]

    # reshaped = np.reshape(corner_points,(18,18))
    # print(reshaped)

    # for x in rows:
    #     for y in cols:
    #         arr[x][y] = corner_points([x, y])



# Select the live video stream source (0-webcam & 1-GoPro)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Show the starting board either as blank or with the initial setup
# start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
# blank = '8/8/8/8/8/8/8/8'
#
# board = chess.Board()
# board.push_san("e4")
# board.push_san("e5")
# print(board.legal_moves)
# print(board.fen())  # prints fen
# print(board.san(2))  # prints san
#
# print(chess.Move.from_uci("e2e4") in board.legal_moves)  # Check Legal Moves
# board = fen_to_image(start)
# board_image = cv2.imread('current_board.png')
# cv2.imshow('current board', board_image)

# Loads model
model = tensor()

# Calibrates the board
calibrate_board(False)

# Run detection
while True:
    save_crop_img()
    print("Running Detection....")
    try:
        classes, boxes, img = main(model)
        print("new boundaries", np.array(trans_boxes(img, boxes)))
    except TypeError:
        print(TypeError)

    # print(classes, boxes)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        calibrate_board(False)
        continue

    if cv2.waitKey(2) & 0xFF == ord('q'):
        # End the program
        break

cap.release()
cv2.destroyAllWindows()
