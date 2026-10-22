import cv2
import os

# 图片路径
IMG_PATH = r"D:\Users\Admin\PycharmProjects\rail_detection_project\dataset\train\images\gettyimages-1231242391-612x612_jpg.rf.b4a54de33ef486817f3d18ecdf658b58.jpg"
# 手动写对应txt完整路径
TXT_PATH = r"D:\Users\Admin\PycharmProjects\rail_detection_project\dataset\train\labels\gettyimages-1231242391-612x612_jpg.rf.b4a54de33ef486817f3d18ecdf658b58.txt"

img = cv2.imread(IMG_PATH)

if img is None:
    print("图片读取失败！")
else:
    h, w = img.shape[:2]
    if not os.path.exists(TXT_PATH):
        print("对应标注txt文件不存在")
    else:
        with open(TXT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 修复：先转列表，只取前5个数值
            nums = list(map(float, line.split()))
            cls, xc, yc, bw, bh = nums[:5]

            # 归一化坐标还原像素框
            xc *= w
            yc *= h
            bw *= w
            bh *= h
            x1 = int(xc - bw / 2)
            y1 = int(yc - bh / 2)
            x2 = int(xc + bw / 2)
            y2 = int(yc + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.imwrite("./label_show.png", img)
        cv2.imshow("数据集标注可视化", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("标注可视化图片已生成在draw文件夹")