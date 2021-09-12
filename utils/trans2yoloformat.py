import os
from PIL import Image

def trans_origin_to_yolo(dir, filename):
    image_dir = dir + "/images/train/"
    origin_label_dir = dir + "/labels/train/"
    yoloformat_label_dir = dir + "/labels_yoloformat/train/"

    image_name = filename + ".jpg"
    origin_label_name = filename + "_mask.txt"
    yoloformat_label_name = filename + ".txt"

    f_origin_label = open(origin_label_dir + origin_label_name)
    lines = f_origin_label.readlines()  # 整行读取
    f_origin_label.close()
    img_pillow = Image.open(image_dir + image_name)
    img_width = img_pillow.width  # 图片宽度
    img_height = img_pillow.height  # 图片高度

    f_yoloformat_label = open(yoloformat_label_dir + yoloformat_label_name, 'w')
    for line in lines:
        rs = line.split(' ')
        if len(rs) > 4:
            classname = rs[0]
            xmin = float(rs[1])
            ymin = float(rs[2])
            xmax = float(rs[3])
            ymax = float(rs[4])
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            x_center_yoloformat = str(x_center / img_width)
            y_center_yoloformat = str(y_center / img_height)
            width_yoloformat = str(width / img_width)
            height_yoloformat = str(height / img_height)
            f_yoloformat_label.write('0' + ' ' + x_center_yoloformat + ' ' + y_center_yoloformat +
                                     ' ' + width_yoloformat + ' ' + height_yoloformat + '\n')
    f_yoloformat_label.close()


if __name__ == '__main__':
    dir = 'E:/实验室/endocv2021/trainData_EndoCV2021_yolo_9_1_V2_new'
    file_list = os.listdir(dir + '/images/train')
    print(file_list)
    for file in file_list:
        name, file_type = file.split('.')
        trans_origin_to_yolo(dir, name)