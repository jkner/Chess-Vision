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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.blur(gray, (1, 1))
    return img, gray_blur


# def image_gray(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_blur = cv2.blur(gray, (1, 1))
#     return gray_blur

# Canny edge detection
def canny_edge(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper, apertureSize=3)
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


# Hierarchical cluster (by euclidean distance) intersection points
def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])


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


# Crop board into separate images and write to folder
def write_crop_images(img, points, img_count=0, folder_path='./Data/raw_data/'):
    num_list = []
    shape = list(np.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= 8:
        range_num = 8
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - 8) - (row * 11)
        num_list.append(range(start, end, -1))

    for row in num_list:
        for s in row:
            # ratio_h = 2
            # ratio_w = 1
            base_len = math.dist(points[s], points[s + 1])
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            img_count += 1
            cv2.imwrite('./Data/raw_data/data_image' + str(img_count) + '.jpeg', cropped)
            # print(folder_path + 'data' + str(img_count) + '.jpeg')
    return img_count


# Crop board into separate images and shows
def x_crop_images(img, points):
    num_list = []
    img_list = []
    shape = list(np.shape(points))
    start_point = shape[0] - 14

    if int(shape[0] / 11) >= 8:
        range_num = 8
    else:
        range_num = int((shape[0] / 11) - 2)

    for row in range(range_num):
        start = start_point - (row * 11)
        end = (start_point - 8) - (row * 11)
        num_list.append(range(start, end, -1))

    for row in num_list:
        for s in row:
            base_len = math.dist(points[s], points[s + 1])
            bot_left, bot_right = points[s], points[s + 1]
            start_x, start_y = int(bot_left[0]), int(bot_left[1] - (base_len * 2))
            end_x, end_y = int(bot_right[0]), int(bot_right[1])
            if start_y < 0:
                start_y = 0
            cropped = img[start_y: end_y, start_x: end_x]
            img_list.append(cropped)
            # print(folder_path + 'data' + str(img_count) + '.jpeg')
    return img_list


# Convert image from RGB to BGR
def convert_image_to_bgr_numpy_array(image_path, size=(224, 224)):
    image = PIL.Image.open(image_path).resize(size)
    img_data = np.array(image.getdata(), np.float32).reshape(*size, -1)
    # swap R and B channels
    img_data = np.flip(img_data, axis=2)
    return img_data


# Adjust image into (1, 224, 224, 3)
def prepare_image(image_path):
    im = convert_image_to_bgr_numpy_array(image_path)

    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68

    im = np.expand_dims(im, axis=0)
    return im


# Changes digits in text to ints
def atoi(text):
    return int(text) if text.isdigit() else text


# Finds the digits in a string
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


# Reads in the cropped images to a list
def grab_cell_files(folder_name='./Data/raw_data/*'):
    img_filename_list = []
    for path_name in glob.glob(folder_name):
        img_filename_list.append(path_name)
    # img_filename_list = img_filename_list.sort(key=natural_keys)
    return img_filename_list


# Classifies each square and outputs the list in Forsyth-Edwards Notation (FEN)
def generate_fen(pred_list):
    # category_reference = {0: 'K', 1: 'Q', 2: 'B', 3: 'N', 4: 'R', 5: 'P', 6: 'k', 7: 'q', 8: 'b', 9: 'n', 10: 'r',
    #                       11: 'p', 12: '1'}
    # # pred_list = []
    # # for filename in img_filename_list:
    # #     img = prepare_image(filename)
    # #     out = model.predict(img)
    # #     top_pred = np.argmax(out)
    # #     pred = category_reference[top_pred]
    # #     pred_list.append(pred)
    #
    # fen = ''.join(pred_list)
    # fen = fen[::-1]
    # fen = '/'.join(fen[i:i + 8] for i in range(0, len(fen), 8))
    # sum_digits = 0
    # for i, p in enumerate(fen):
    #     if p.isdigit():
    #         sum_digits += 1
    #     elif p.isdigit() is False and (fen[i - 1].isdigit() or i == len(fen)):
    #         fen = fen[:(i - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1)) + fen[i:]
    #         sum_digits = 0
    # if sum_digits > 1:
    #     fen = fen[:(len(fen) - sum_digits)] + str(sum_digits) + ('D' * (sum_digits - 1))
    # fen = fen.replace('D', '')
    # return fen

    # Use StringIO to build string more efficiently than concatenating
    with io.StringIO() as s:
        for row in pred_list:
            empty = 0
            for cell in row:
                c = cell[0]
                if c in ('w', 'b'):
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(cell[1].upper() if c == 'w' else cell[1].lower())
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        # Move one position back to overwrite last '/'
        s.seek(s.tell() - 1)
        # If you do not have the additional information choose what to put
        s.write(' w KQkq - 0 1')
        return s.getvalue()


# Converts the FEN into a PNG file
def fen_to_image(fen):
    board = chess.Board(fen)
    current_board = chess.svg.board(board=board, size=1000)

    output_file = open('current_board.svg', "w")
    output_file.write(current_board)
    output_file.close()

    svg = svg2rlg('current_board.svg')
    renderPM.drawToFile(svg, 'current_board.png', fmt="PNG")

    board_image = cv2.imread('current_board.png')
    cv2.imshow('current board', board_image)


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
        c = True
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
    cropped_img = img[15:1000, 65:590]  # TODO: replace with actual points from the board later

    return cropped_img


# def make_square(im, min_size=416, fill_color=(0, 0, 0, 0)):
#     x, y = im.size
#     print(x, y)
#     size = max(416, x, y)
#     new_im = Image.new('RGBA', (size, size), fill_color)
#     new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
#     print(new_im.size)
#     return new_im


def trans_boxes(img, boxes):
    new_arr = []
    for i in range(len(boxes)):
        x1y1 = (boxes[i][0:2] * img.shape[0:2][::-1]).astype(np.int)
        # print("X1Y1:", x1y1)
        x2y2 = (boxes[i][2:4] * img.shape[0:2][::-1]).astype(np.int)
        new_arr.append(x1y1)
        new_arr.append(x2y2)

    return new_arr


def mid_point(arr):
    # print("point 1", arr[0][0])
    # print("point 1", arr[1][0])
    # (arr[1][0] - arr[0][0])/2 # x values
    # arr[1][1] - (arr[1][1] - arr[0][1])/3

    midpoint_list = list()

    for i in range(0, len(arr), 2):
        x = (arr[i + 1][0] + arr[i][0]) // 2
        y = arr[i + 1][1] - (arr[i + 1][1] - arr[i][1]) // 3
        midpoint = [x, y]
        midpoint_list.append(midpoint)
        # print("midpoint list", midpoint_list)

    # print("midpoint list", midpoint_list)
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

    cv2.imshow("transformed image", warped_img)

    return warped_img


# Draws midpoints on the warped image:
def draw_boundary_warp(warped_img, midpoints):
    for mid in midpoints:
        warped_img = cv2.circle(warped_img, (int(mid[0]), int(mid[1])), 2, (255, 0, 0), 2)



def classify_2d(classify_arr, predict_arr):
    category_reference = {0: 'wk', 1: 'wq', 2: 'wb', 3: 'wn', 4: 'wr', 5: 'wp', 6: 'bk', 7: 'bq', 8: 'bb', 9: 'bn',
                          10: 'br', 11: 'bp', 12: 'em'}

    predicted_list = np.empty(shape=(8, 8), dtype=object)
    predicted_list.fill('em')

    for classify, predict in zip(classify_arr, predict_arr):
        predicted_list[classify[0]][classify[1]] = category_reference[predict]

    flipped_arr = np.fliplr(predicted_list)
    # print("flipped_arr", flipped_arr)
    # transposed_list = predicted_list.T
    #
    # flatten_list = transposed_list.flatten()

    return flipped_arr


# classifies using python-chess object notation:

def classify_object_notation(classify_arr, predict_arr):
    category_reference = {0: 'K', 1: 'Q', 2: 'B', 3: 'N', 4: 'R', 5: 'P', 6: 'k', 7: 'q', 8: 'b', 9: 'n',
                          10: 'r', 11: 'p', 12: '.'}

    pred_list = np.empty(shape=(8, 8), dtype=object)
    pred_list.fill('.')

    for classify, predict in zip(classify_arr, predict_arr):
        pred_list[classify[0]][classify[1]] = category_reference[predict]

    flipped_arr = np.fliplr(pred_list)
    # print("flipped_arr", flipped_arr)

    row1 = flipped_arr[0]
    row1 = (','.join(row1))  # remove commas
    row1 = row1.replace(',', ' ')  # replace commas with space
    row1 = str(row1)

    new_arr = []

    for row in flipped_arr:
        new_row = (','.join(row)).replace(',', ' ')
        new_arr.append(new_row)

    # print(new_arr)
    # print("row1", row1)
    return new_arr


def convert_cell(value):
    if value == 'em':
        return None
    else:
        color, piece = value
        return piece.upper() if color == 'w' else piece.lower()


def convert_rank(rank):
    return ''.join(
        value * count if value else str(count)
        for value, count in run_length.encode(map(convert_cell, rank))
    )


def fen_from_board(board):
    return '/'.join(map(convert_rank, board)) + ' w KQkq - 0 1'


def get_uci(board1, board2, who_moved):
    nums = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h"}
    str_board = str(board1).split("\n")
    # print("strboard split", str_board )
    # print("strboard split2", board2)
    move = ""
    flip = False
    if who_moved:  # If true then it's white's turn
        for i in range(8)[::-1]:
            for x in range(15)[::-1]:
                # print("strboard1", str_board[i][x])
                # print("strboard2", board2[i][x] )
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
    print("WHO MOVED:", who_moved)
    print("MOVE: ", move)
    if move == 'h1g1f1e1':
        # return 'h1g1'
        return 'e1g1'
    elif move == 'e8f8g8h8':
        return 'e8g8'
    elif move == 'e1d1c1a1':
        return 'e1c1'
        # return 'e1c1'
    elif move == 'e8d8c8a8':
        return 'e8c8'
    return move


# Casesg
# Capture
# Movement
# Castling
# Promotion
#
# def fen_compare(pred_list1, pred_list2):
#
#     return move


def fen_to_pil(fen):
    pil_image = draw.transform_fen_pil(
        fen=fen,
        board_size=480,
        light_color=(255, 253, 208),
        dark_color=(0, 127, 70)
    )
    # pil_image.show()
    # cv2.imwrite("new_board.png", pil_image)
    board_image = cv2.imread('new_board.png')
    # cv2.imshow("current_board", pil_image)
    # renderPM.drawToFile(pil_image, 'current_board.png', fmt="PNG")
    #
    #
    # board_image = cv2.imread('current_board.png')
    # cv2.imshow('current board', board_image)
    open_cv_image = np.array(pil_image.convert('RGB'))
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    cv2.imshow("current_board", open_cv_image)
