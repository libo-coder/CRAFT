# coding=utf-8
import numpy as np

def cvt_boxes(boxes):
    new_boxes = []
    for box in boxes:
        new_box = {}
        new_box['cx'] = (box['locale']['left_top'][0] + box['locale']['right_bottom'][0]) / 2
        new_box['cy'] = (box['locale']['left_top'][1] + box['locale']['right_bottom'][1]) / 2
        new_box['w'] = box['locale']['right_bottom'][0] - box['locale']['left_top'][0] + 1
        new_box['h'] = box['locale']['right_bottom'][1] - box['locale']['left_top'][1] + 1
        new_box['d'] = 0
        new_boxes.append(new_box)
    return new_boxes


def cvt_boxes2(boxes):
    new_boxes = []
    for box in boxes:
        new_box = {}
        new_box['words'] = box['text']
        new_box['locale'] = {'left_top': [box['cx'] - box['w'] / 2, box['cy'] - box['h'] / 2],
                             'right_bottom': [box['cx'] + box['w'] / 2, box['cy'] + box['h'] / 2]}
        new_boxes.append(new_box)
    return new_boxes


def mg_boxes(boxes, T=1, also=False):
    new_boxes = []
    for cur_box in boxes:
        if len(new_boxes) == 0:
            new_boxes.append([cur_box])
        else:
            check = False
            for pre_box in new_boxes[-1]:
                if diff_boxes(cur_box, pre_box, T, also):
                    check = True
            if check:
                new_boxes.append([cur_box])
            else:
                new_boxes[-1].append(cur_box)
    return [sort_group(group) for group in new_boxes]


def sort_group(boxes):
    boxes = sorted(boxes, key=lambda box: box['cx'])

    tmp_boxes = []
    for box in boxes:
        tmp_boxes.append([box['cx'] - box['w'] / 2, box['cy'] - box['h'] / 2,
                          box['cx'] + box['w'] / 2, box['cy'] - box['h'] / 2,
                          box['cx'] + box['w'] / 2, box['cy'] + box['h'] / 2,
                          box['cx'] - box['w'] / 2, box['cy'] + box['h'] / 2])

    tmp_boxes = np.array(tmp_boxes)
    ltx = tmp_boxes[:, 0].min()
    lty = tmp_boxes[:, 1].min()
    rbx = tmp_boxes[:, 4].max()
    rby = tmp_boxes[:, 5].max()

    return {'cx': (ltx + rbx) / 2, 'cy': (lty + rby) / 2, 'w': rbx - ltx + 1, 'h': rby - lty + 1, 'd': 0}


# y差的大，真
# y差的小，但横向距离差的大，真
def diff_boxes(box1, box2, T, also):
    flag = abs(box1['cy']-box2['cy']) / max(0.01, min(box1['h']/2, box2['h']/2)) > T
    # 如果两个里有任意一个字数少，也应该返回假（合并）
    if (not flag) and also:
        flag = abs(box1['cx']-box2['cx']) > (box1['w']/2+box2['w']/2) * (1+3*T)
    return flag