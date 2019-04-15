import numpy as np
import os
import cv2

from chainercv.utils import read_image
from chainercv.datasets.voc import voc_utils
from chainercv.chainer_experimental.datasets.sliceable import GetterDataset

class ConvertLabelToInstanceDataset(GetterDataset):

    def __init__(self, list_path='list\\train.txt', label_colors=None, image_dir='image', image_ext='.png', label_dir='label', label_ext='.png'):

        super(ConvertLabelToInstanceDataset, self).__init__()

        self.list_path = list_path
        self.ids = [id_.strip() for id_ in open(list_path)]

        self.label_colors = label_colors

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_ext = image_ext
        self.label_ext = label_ext

        self.add_getter('img', self._get_image)
        self.add_getter(('mask', 'label'), self._get_annotations)

    # データサイズを取得
    def __len__(self):
        data_num = len(self.ids)
        
        return data_num

    # 入力画像を取得
    def _get_image(self, i):
        # 入力画像を読込み
        data_id = self.ids[i]
        image_file_path = os.path.join(self.image_dir, data_id + self.image_ext)

        img = read_image(image_file_path, color=True)

        return img

    # アノテーション（マスク画像、ラベル）を取得
    def _get_annotations(self, i):
        data_id = self.ids[i]

        # ラベル画像をインスタンス画像に変換
        label_img, inst_img = self._load_label_convert_inst(data_id)

        # 画像変換
        mask_img, label = voc_utils.image_wise_to_instance_wise(label_img, inst_img)

        # mask_img, label
        # 入力画像（3xHxW）
        # マスク画像（インスタンス数xHxW、bool）
        # ラベル（各インスタンス数、各インスタンスのラベル番号）
        return mask_img, label

    # ラベル画像を読込み、インスタンス画像に変換
    def _load_label_convert_inst(self, data_id):

        # ラベル画像を読込み
        label_file_path = os.path.join(self.label_dir, data_id + self.label_ext)
        label_img = read_image(label_file_path, color=True)

        # インスタンス数
        all_instance_num = 0

        # 出力変数
        # ラベル画像（変換後）・インスタンス画像（変換後）
        # label_img_out = np.full_like(label_img[0], -1, dtype=np.int32)
        label_img_out = np.full_like(label_img[0], 0, dtype=np.int32)
        instance_img_out = np.full_like(label_img[0], -1, dtype=np.int32)

        # クラス毎
        class_num = len(self.label_colors)
        for class_id in range(1, class_num):

            # 対象クラスの画素を取得
            class_img = np.zeros_like(label_img, dtype=np.bool)
            class_img[0] = (label_img[0] == self.label_colors[class_id, 0])
            class_img[1] = (label_img[1] == self.label_colors[class_id, 1])
            class_img[2] = (label_img[2] == self.label_colors[class_id, 2])
            class_img = np.logical_and(np.logical_and(class_img[0], class_img[1]),class_img[2])

            # クラス番号保存
            label_img_out[class_img == True] = class_id
            class_img = class_img.astype(np.uint8)

            # 粒子解析
            _, contours, _ = cv2.findContours(class_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            instance_num = len(contours)
            for instance_id in range(instance_num):
                contour = contours[instance_id]

                # 対象インスタンスの画素を取得
                instance_img = np.zeros_like(label_img[0])
                instance_img = cv2.drawContours(instance_img, [contour], 0, 1, -1) 

                # インスタンス番号保存
                instance_img_out[instance_img == 1] = all_instance_num + instance_id

            all_instance_num += instance_num

        # label_img_out, instance_img_out
        # ラベル画像（HxW、各画素にクラス番号（0～最大クラス番号-1））
        # インスタンス画像（HxW、各画素にインスタンス番号（0～最大インスタンス番号-1））
        return label_img_out, instance_img_out
