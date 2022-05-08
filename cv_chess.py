import re
import time

import cv2
import numpy as np
import chess
import chess.pgn

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
                                trans_boxes,
                                mid_point,
                                avg,
                                classify_squares,
                                perspective_transform,
                                get_perspective_transform,
                                board_corners,
                                adjust_classifier,
                                draw_boundary_warp,
                                warp_transform,
                                classify_2d,
                                generate_fen,
                                fen_from_board,
                                get_uci,
                                classify_object_notation,
fen_to_pil
                                )

from detect import (main,
                    tensor)


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
    #print("num of corners", num_of_corners)
    return drawn, inner_points2


def save_crop_img():
    ret, frame = cap.read()
    cv2.imshow('live', frame / 255)
    out = cv2.addWeighted(frame, 1, frame, 0, 1)
    cv2.imwrite('frame.jpeg', out)
    crop_frame = crop_image(out)
    cv2.imwrite('./images/chess_pictures/cropped_frame.jpeg', crop_frame)
    return crop_frame


# Calibrate board
def calibrate_board(calibrated):
    while not calibrated:
        print('Calibrating Board....')
        save_crop_img()
        if cv2.waitKey(0):
            #print('C....')
            img, corner_points = detect_corners()
            if cv2.waitKey(0) & 0xFF == ord('s'):
                # if len(corner_points) == 81:
                print("saving corner image...")
                cv2.imwrite('corner.jpeg', img)
                corner_array = save_corner_points(corner_points)
                #print("2d Array", corner_array)
                return corner_array, corner_points


def save_corner_points(corner_points):
    arr = np.zeros((9, 9, 2), dtype=int)
    array_int = np.array(corner_points).astype(int)

    #print(array_int)

    i = 0

    for row_index, row in enumerate(arr):
        for col_index, item in enumerate(row):
            arr[row_index][col_index] = array_int[i]
            #print(row_index, col_index, i, array_int[i], arr[row_index][col_index], end=" ")
            if i == 80:
                #print("return arr")
                #print("returned array", arr)
                return arr

            i += 1
   # print("arr", arr)


# Select the live video stream source (0-webcam & 1-GoPro)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Show the starting board either as blank or with the initial setup
start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
new_move1 = ''
new_move2 = ''

board2 = chess.Board()
#board2.push_san("e4")

# blank = '8/8/8/8/8/8/8/8'
#
board = chess.Board()

# board.push_san("e2e4")
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
board_array, corner_fp = np.array(calibrate_board(False))
# print("Board Array", board_array)
# print("first item", board_array[0][0])
# print("first item", board_array[1][1])
# print("first item", board_array[0][0])
# print("first item", board_array[0][0])
# print("last item", board_array[8][8])

corners = board_corners(board_array)
#print("corners", corners)

start_turn = "w"

game = chess.pgn.Game()
game.headers["Event"] = "Example"

move_arr = []

# Run detection
while True:
    cropped_image = save_crop_img()
    print("Running Detection....")
    try:
        classes, boxes, img = main(model)
        boundary_arr = np.float32(trans_boxes(img, boxes))
        mid_array = mid_point(boundary_arr)
        #print("new boundaries", boundary_arr)
        transform = get_perspective_transform(corners, cropped_image)
        # boundary_points_transform = perspective_transform(boundary_arr)
        boundary_points_transform = perspective_transform(mid_array, transform)

        corner_transform = perspective_transform(board_array, transform)

        warped_img = warp_transform(cropped_image, transform)

        draw_boundary_warp(warped_img, boundary_points_transform)
        #print("corner transform", corner_transform)

        #print("boundary points transform", boundary_points_transform)

        # size = avg(board_array)
        classify_arr = classify_squares(51.5, boundary_points_transform)

        prediction_list = classify_2d(classify_arr, classes)

        new_board = classify_object_notation(classify_arr, classes)

        fen = fen_from_board(prediction_list)

        #print("FEN", fen)

        #fen_to_image(fen)

        new_move = get_uci(board, new_board, start_turn)

        print("new_move", new_move)

        try:
            valid = chess.Move.from_uci(new_move) in board.legal_moves
            print("BOARD", board)
            print("VALID", valid)

            if valid:
                board.push_san(str(new_move))
                move_arr.append(new_move)
                #print("BOARD", board)
                #print("MOVES: ", board2.variation_san([chess.Move.from_uci(m) for m in move_arr]))

                #fen_to_image(board.fen())
                fen_to_pil(board.fen())
                if start_turn == "w":
                    start_turn == "b"
                else:
                    start_turn == "w"

            else:
                continue
            print("MOVES: ", board2.variation_san([chess.Move.from_uci(m) for m in move_arr]))
        except ValueError:
            print("Did not Detect Valid Move")
        print("MOVES: ", board2.variation_san([chess.Move.from_uci(m) for m in move_arr]))

        #print("MOVES: ", board.variation_san([chess.Move.from_uci(m) for m in move_arr]))
        # adjust_classifier(mid_array, classify_arr, board_array)
        # print("GAME.MOVE: ", game.move)
        # print("PLY", game.ply())
        #
        # print("BOARD PLY", board.ply())
        # print("PGN:", game)
        # print("current board:", board)
    except:
        print("ERROR")

    # print(classes, boxes)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        calibrate_board(False)
        continue

    if cv2.waitKey(2) & 0xFF == ord('q'):
        # End the program
        break

cap.release()
cv2.destroyAllWindows()
