import math
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean
import chess
import chess.svg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import re
import glob
import PIL
import io
from more_itertools import run_length
from fen2pil import draw


# Read image and do lite image processing
def read_img(file):
    img = cv2.imread(str(file))
    out = cv2.addWeighted(img, 1, img, 0, 0)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (1, 1))
    return img, gray_blur


# Canny edge detection
def canny_edge(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper, apertureSize=3)

    cv2.imshow("1", edges)
    return edges


# Hough line detection
def hough_line(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
    lines = np.reshape(lines, (-1, 2))
    # cv2.imwrite("testHough.jpg", lines)
    return lines


# Separate line into horizontal and vertical
def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])

    return h_lines, v_lines


# Find the intersections of the lines
def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)


# Average the y value in each row and augment original points
def augment_points(points):
    points_shape = list(np.shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 11)):
        start = row * 11
        end = (row * 11) + 10
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = mean(rw_y)
        for i in range(len(rw_x)):
            point = (rw_x[i], y_mean)
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points


def dist(i, p):  # finds distance between pts, kinda
    # return ((i[0] - p[0])**2 + (i[1] - p[1])**2)**0.5
    res = (abs(i[0] - p[0]) + abs(i[1] - p[1]))
    if res == 0:
        return 100
    return res


def remove_duplicates(list_of_points):
    # list = set(list)
    # return [p for p in list if all(dist(i,p) > 3 for i in list)]
    out = []
    for i in range(len(list_of_points)):
        if all(dist(list_of_points[i], p) > 15 for p in out):
            out.append(list_of_points[i])
    return np.array(out)


def remove_outside_points(list_of_points):
    out = []
    excluded_val = [21, 22, 32, 33, 43, 44, 54, 55, 65, 66, 76, 77, 87, 88, 98, 99, 109, 110]
    for i in range(len(list_of_points) - 11):
        if i in excluded_val:
            pass
        else:
            if i > 11:
                out.append(list_of_points[i])

    return np.array(out)


def crop_image(img):
    cropped_img = img[15:1000, 65:590]

    return cropped_img


def trans_boxes(img, boxes):
    new_arr = []
    for i in range(len(boxes)):
        x1y1 = (boxes[i][0:2] * img.shape[0:2][::-1]).astype(np.int)
        x2y2 = (boxes[i][2:4] * img.shape[0:2][::-1]).astype(np.int)
        new_arr.append(x1y1)
        new_arr.append(x2y2)

    return new_arr


def mid_point(arr):
    midpoint_list = list()

    for i in range(0, len(arr), 2):
        x = (arr[i + 1][0] + arr[i][0]) // 2
        y = arr[i + 1][1] - (arr[i + 1][1] - arr[i][1]) // 3
        midpoint = [x, y]
        midpoint_list.append(midpoint)

    return midpoint_list


def avg(arr):
    x = 0
    y = 0
    for i in range(0, 7):
        x += (arr[i + 1][i + 1][0] - arr[i][i][0]) // 8
        y += (arr[i + 1][i + 1][1] - arr[i][i][1]) // 8

    size = [x, y]
    # print("size", size)
    return size


def classify_squares(size, midpoint):
    classify_arr = []
    for i in midpoint:
        # print("Midpoint", i[0])

        x = i[0]
        y = i[1]

        square_x = x // size
        square_y = y // size

        classify_arr.append([int(square_x), int(square_y)])

        # print("square x", square_x, "square_y", square_y)

    # print("classify arr", classify_arr)
    return classify_arr


def board_corners(arr):
    rect = np.float32([arr[0][0], arr[0][8], arr[8][0], arr[8][8]])
    # rect = np.array([list(arr[0]), list(arr[8]), list(arr[71]), list(arr[80])])
    # print("RECT", rect)
    return rect


# Performs perspective transform on board corners
def get_perspective_transform(corners, img):
    height = 412
    width = 412
    dst = np.array([[0, 0], [width, 0], [0, width], [height, width]], dtype="float32")
    transform = cv2.getPerspectiveTransform(corners, dst)
    # print("transform points", transform)

    return transform


# Performs perspective transform on midpoints
def perspective_transform(mid_points, shape):

    transformed_midpoint = []
    #transformed_midpoint = np.empty(dtype=object)

    for i in mid_points:
        new_arr = [i[0], i[1], 1]
        x, y, z = shape.dot(new_arr)
        # transf_homg_point /= transf_homg_point[2]
        transformed_midpoint.append([x // z, y // z])

    # print("transformed midpoint", transformed_midpoint)
    return transformed_midpoint


# Warps the image using the perspective transform:
def warp_transform(img, transform):
    height = 412
    width = 412
    warped_img = cv2.warpPerspective(img, transform, (width, height))
    #cv2.imshow("transformed image", warped_img)

    return warped_img


# Draws midpoints on the warped image:
def draw_boundary_warp(warped_img, midpoints):
    for mid in midpoints:
        warped_img = cv2.circle(warped_img, (int(mid[0]), int(mid[1])), 2, (255, 0, 0), 2)

    cv2.imshow("transformed image", warped_img)


def classify_2d(classify_arr, predict_arr):
    category_reference = {0: 'wk', 1: 'wq', 2: 'wb', 3: 'wn', 4: 'wr', 5: 'wp', 6: 'bk', 7: 'bq', 8: 'bb', 9: 'bn',
                          10: 'br', 11: 'bp', 12: 'em'}

    predicted_list = np.empty(shape=(8, 8), dtype=object)
    predicted_list.fill('em')

    try:

        for classify, predict in zip(classify_arr, predict_arr):
            predicted_list[classify[0]][classify[1]] = category_reference[predict]

        flipped_arr = np.fliplr(predicted_list)

        return flipped_arr

    except IndexError:
        pass


# classifies using python-chess object notation:
def classify_object_notation(classify_arr, predict_arr):
    category_reference = {0: 'K', 1: 'Q', 2: 'B', 3: 'N', 4: 'R', 5: 'P', 6: 'k', 7: 'q', 8: 'b', 9: 'n',
                          10: 'r', 11: 'p', 12: '.'}

    prediction_list = np.empty(shape=(8, 8), dtype=object)
    prediction_list.fill('.')

    try:

        for classify, predict in zip(classify_arr, predict_arr):
            prediction_list[classify[0]][classify[1]] = category_reference[predict]

        flipped_arr = np.fliplr(prediction_list)

        new_arr = []

        for row in flipped_arr:
            new_row = (','.join(row)).replace(',', ' ')
            new_arr.append(new_row)

        return new_arr
    except IndexError:
        pass


def get_uci(board1, board2, who_moved):
    nums = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h"}
    str_board = str(board1).split("\n")
    move = ""
    flip = False

    if who_moved:  # If true then it's white's turn
        for i in range(8)[::-1]:
            for x in range(15)[::-1]:
                if str_board[i][x] != board2[i][x]:
                    if str_board[i][x] == "." and move == "":
                        flip = True
                    move += str(nums.get(round(x / 2) + 1)) + str(9 - (i + 1))
    else:
        for i in range(8):
            for x in range(15):
                if str_board[i][x] != board2[i][x]:
                    if str_board[i][x] == "." and move == "":
                        flip = True
                    move += str(nums.get(round(x / 2) + 1)) + str(9 - (i + 1))
    if flip:
        move = move[2] + move[3] + move[0] + move[1]

    # Checks if a move is castling:
    if move == 'h1g1f1e1':
        # return 'h1g1'
        return 'e1g1'
    elif move == 'e8f8g8h8':
        return 'e8g8'
    elif move == 'e1d1c1a1':
        return 'e1c1'
        # return 'e1c1'
    elif move == 'a8c8d8e8':
        return 'e8c8'
    return move


def fen_to_pil(fen):
    pil_image = draw.transform_fen_pil(
        fen=fen,
        board_size=480,
        light_color=(255, 253, 208),
        dark_color=(0, 127, 70)
    )

    open_cv_image = np.array(pil_image.convert('RGB'))
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cv2.imshow("current_board", open_cv_image)
