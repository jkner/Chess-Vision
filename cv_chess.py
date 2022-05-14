import re
import time
from datetime import date
from typing import Union

import chess.pgn
import cv2
import numpy as np

from cv_chess_functions import (read_img,
                                canny_edge,
                                hough_line,
                                h_v_lines,
                                line_intersections,
                                augment_points,
                                crop_image,
                                remove_duplicates,
                                remove_outside_points,
                                trans_boxes,
                                mid_point,
                                classify_squares,
                                perspective_transform,
                                get_perspective_transform,
                                board_corners,
                                draw_boundary_warp,
                                warp_transform,
                                classify_2d,
                                get_uci,
                                classify_object_notation,
                                fen_to_pil
                                )
from detect import (main,
                    tensor)


# Resize the frame by scale by dimensions
def rescale_frame(frame):
    dim = (416, 416)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Finding Corners
def detect_corners():
    print("Detecting corners...")

    # Low-level CV techniques (grayscale & blur)
    img, gray_blur = read_img('./images/chess_pictures/cropped_frame.jpeg')

    # Canny algorithm
    edges = canny_edge(gray_blur)

    # Hough Transform
    lines = hough_line(edges)

    # Separate the lines into vertical and horizontal lines
    h_lines, v_lines = h_v_lines(lines)

    # Find and cluster the intersecting
    intersection_points = line_intersections(h_lines, v_lines)

    # Combine cluster into one point
    cluster = remove_duplicates(intersection_points)

    # Final coordinates of the board
    aug_points = np.array(augment_points(cluster))

    # Remove Outside Points:
    inner_points = remove_outside_points(aug_points)

    # Draws Chessboard Corners using corner points
    drawn = cv2.drawChessboardCorners(img, (9, 9), inner_points, True)

    cv2.imwrite('./images/board_images/board_with_corners.jpeg', drawn)
    cv2.imshow('Corners', drawn)

    return drawn, inner_points


def save_crop_img():
    ret, frame = cap.read()
    cv2.imshow('live', frame / 255)
    crop_frame = crop_image(frame)
    cv2.imwrite('./images/chess_pictures/cropped_frame.jpeg', crop_frame)
    return crop_frame


# Calibrate board
def calibrate_board(calibrated):
    while not calibrated:
        print('Calibrating Board...')
        save_crop_img()
        if cv2.waitKey(0):
            img, corner_points = detect_corners()
            if cv2.waitKey(0) & 0xFF == ord('s'):
                # if len(corner_points) == 81:
                print("saving corner image...")
                cv2.imwrite('./images/board_images/corner.jpeg', img)
                corner_array = save_corner_points(corner_points)
                # print("2d Array", corner_array)
                return corner_array, corner_points


# Save corner points in a 9x9 matrix
def save_corner_points(corner_points):
    arr = np.zeros((9, 9, 2), dtype=int)
    array_int = np.array(corner_points).astype(int)
    # print(array_int)

    i = 0
    for row_index, row in enumerate(arr):
        for col_index, item in enumerate(row):
            arr[row_index][col_index] = array_int[i]
            # print(row_index, col_index, i, array_int[i], arr[row_index][col_index], end=" ")
            if i == 80:
                # print("return arr")
                # print("returned array", arr)
                return arr
            i += 1


# Select the live video stream source (0-webcam & 1-GoPro)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Show the starting board either as blank or with the initial setup
start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

# Boards used for move detection and PGN Generation
new_board = chess.Board()
board = chess.Board()

# Loads model
model = tensor()

# Calibrates the board
board_array, corner_fp = calibrate_board(False)
corners = board_corners(board_array)

today = date.today()
now = int(time.time())

# Move stack
move_arr = []

# Create a new PGN File:
new_pgn = open("./pgn/unix-" + str(now) + ".pgn", "x")

# PGN Header
event = input("Enter the event name: ")
site = input("Enter the site of the event: ")
#pgndate = input("Enter the date: ")
round = input("Enter the round: ")
playerwhite = input("Enter the name of the player of the white pieces: ")
playerblack = input("Enter the name of the player of the black pieces: ")
#String
eventString = "[Event \"" + event + "\"]"
siteString = "\n[Site \"" + site + "\"]"
pgndateString = "\n[Date \"" + str(today) + "\"]"
roundString = "\n[Round \"" + round + "\"]"
playerwhiteString = "\n[White \"" + playerwhite + "\"]"
playerblackString = "\n[Black \"" + playerblack + "\"]\n"
#Output to file
new_pgn.write(eventString)
new_pgn.write(siteString)
new_pgn.write(pgndateString)
new_pgn.write(roundString)
new_pgn.write(playerwhiteString)
new_pgn.write(playerblackString)

# Run detection
print("Running Detection....")
while True:
    cropped_image = save_crop_img()

    try:
        classes, boxes, img = main(model)
        # transforms boundary box

        boundary_arr = np.float32(trans_boxes(img, boxes))
        # finds the mid-point of each boundary box

        mid_array = mid_point(boundary_arr)
        # creates a transform matrix based on the corners
        transform = get_perspective_transform(corners, cropped_image)

        # transforms the boundary box using transform matrix
        boundary_points_transform = perspective_transform(mid_array, transform)

        # transforms the corners of the chessboard using transform matrix
        corner_transform = perspective_transform(board_array, transform)

        # show a warped image using transform matrix
        warped_img = warp_transform(cropped_image, transform)

        # draw points on the warped image
        #draw_boundary_warp(warped_img, boundary_points_transform)

        # add each midpoint to a 8x8 matrix to determine its location on the board
        classify_arr = classify_squares(51.5, boundary_points_transform)

        # combine location of the detected pieces and the classes
        prediction_list = classify_2d(classify_arr, classes)

        # same as prediction list but convert to the same format as python chess library board
        new_frame = classify_object_notation(classify_arr, classes)

        # generate a move from the previous board state and current board
        new_move = get_uci(board, new_frame, board.turn)

        # show fen image of the last valid board
        fen_to_pil(board.fen())

        print("Current Move: ", new_move)

        # Move event detector:
        try:
            # check if a move is valid
            valid = chess.Move.from_uci(new_move) in board.legal_moves

            # if it is valid, add push the move to both the board and move stack
            if valid:
                print("Valid move:", new_move)
                board.push_san(str(new_move))
                move_arr.append(new_move)
                print("MOVES: ", new_board.variation_san([chess.Move.from_uci(m) for m in move_arr]))

                # if it is checkmate, end the game
                if board.is_checkmate():
                    if board.outcome():
                        new_pgn.write('[Result "1-0"]\n')
                        new_pgn.write(new_board.variation_san([chess.Move.from_uci(m) for m in move_arr]))
                        new_pgn.close()
                    else:
                        new_pgn.write('[Result "0-1"]\n')
                        new_pgn.write(new_board.variation_san([chess.Move.from_uci(m) for m in move_arr]))
                        new_pgn.close()

                    break

                # if it is stalemate, end the game
                if board.is_stalemate():
                    new_pgn.write('[Result "1/2-1/2"]')
                    new_pgn.write(new_board.variation_san([chess.Move.from_uci(m) for m in move_arr]))
                    new_pgn.close()
                    break

            else:
                continue

        except ValueError:
            pass

    except TypeError:
        pass

    # recalibrate board
    if cv2.waitKey(1) & 0xFF == ord('c'):
        calibrate_board(False)
        continue

    # quit and save the game
    if cv2.waitKey(2) & 0xFF == ord('q'):
        new_pgn.write(new_board.variation_san([chess.Move.from_uci(m) for m in move_arr]))
        new_pgn.close()
        # End the program
        break

cap.release()
cv2.destroyAllWindows()
