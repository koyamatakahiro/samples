import numpy as np
import matplotlib.pyplot as plt

from convert_label_to_instance_dataset import ConvertLabelToInstanceDataset

def main():
    # パラメータ
    list_path = 'list\\test.txt'
    label_colors = np.array([[255, 255, 255], [0, 0, 255], [255, 0, 0]])

	# dataset作成
    dataset_sample = ConvertLabelToInstanceDataset(list_path=list_path, label_colors=label_colors)

    # データ数
    data_num = dataset_sample.__len__()
    # テスト
    print('data num : ', data_num)

    for i in range(data_num):

        # 画像
        img = dataset_sample._get_image(i)
        # テスト
        # print('img.shape : ', img.shape)
        # img_debug = img.transpose(1,2, 0).astype('uint8')
        # plt.imshow(img_debug)
        # plt.show()

        # アノテーション
        mask_img, label = dataset_sample._get_annotations(i)
        # テスト
        # print('mask_img.shape : ', mask_img.shape)
        # print('label : ', label)

        # テスト
        print('i : {}, instance : {}, blue : {}, red : {} '.format(i, len(label), np.sum(label == 0), np.sum(label == 1)))

if __name__ == '__main__':
    main()
