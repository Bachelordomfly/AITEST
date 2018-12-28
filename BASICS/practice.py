import cv2
import numpy as np

img1 = cv2.imread('20.jpg')
# img2 = cv2.imread('21.jpg', 0)
# cv2.namedWindow('window1')
# cv2.namedWindow('window2')
# cv2.imshow('window1', img1)
# cv2.imshow('window2', img2)
# img3 = img1.copy()
# cv2.imwrite('new_img.jpg', img3, [cv2.IMWRITE_JPEG_QUALITY, 2])
# # cv2.imwrite('rose_copy1.jpg', img1, [cv2.IMWRITE_JPEG_QUALITY, 2])
# img4 = cv2.imread('new_img.jpg')
# cv2.namedWindow('window3')
# cv2.imshow('window3', img4)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#   像素赋值
def white(img, numbers):
    for i in range(numbers):
        for j in range(numbers):
            for k in range(3):
                img[i][j][k] = 255
    return img

#   椒盐函数
def salt(img, numbers):
    for x in range(numbers):
        i = np.random.randint(img.shape[0])
        j = np.random.randint(img.shape[1])
        for k in range(3):
            img[i][j][k] = 255
    return img

#   通道分离
def conduit(img):

    b, g, r = cv2.split(img)
    pic = np.zeros(np.shape(img), np.uint8)  # 像素清零
    pic[:, :, 0] = b
    cv2.namedWindow('Blue')
    cv2.imshow('Blue', pic)
    cv2.waitKey(0)

    pic[:, :, 1] = g
    cv2.namedWindow('Green')
    cv2.imshow('Green', pic)
    cv2.waitKey(0)

    pic[:, :, 2] = r
    cv2.namedWindow('Red')
    cv2.imshow('Red', pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#   通道合并
def merge(b, g, r):
    m = cv2.merge([b, g, r])    # cv2自带通道合并方法
    dstack = np.dstack([b, g, r])    # np的dstack

#   画图
def drew():
    pic = np.zeros([512, 512, 3])
    cv2.line(pic, (256, 512), (256, 0), (255, 255, 0), 4)  # 直线（起点坐标和终点坐标）
    cv2.rectangle(pic, (128, 384), (384, 128), (0, 255, 255), 4)  # 矩形（左上角坐标和右下角坐标）
    cv2.circle(pic, (256, 256), 50, (250, 250, 250), 4)  # 圆（圆心坐标和半径）
    cv2.ellipse(pic, (256, 256), (128, 64), 90, 0, 360, (255, 0, 255), 4)  # 椭圆（圆心坐标，（长半径，短半径），逆时针旋转角度，逆时针开始画图的角度， 逆时针结束画图角度）
    ply = np.array([[50, 190], [380, 420], [255, 50], [120, 420], [450, 190]])  # 注：像素数组
    cv2.polylines(pic, [ply], True, (200, 100, 0), 4)  # 多边形（像素数组，是否封口）
    cv2.putText(pic, 'this is change', (128, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (100, 100, 100), 2)  # 文字（text,起始坐标，字体，文字大小，颜色，线宽）
    cv2.namedWindow('line')
    cv2.imshow('line', pic)
    cv2.waitKey(0)
    cv2.destroyWindow('line')

#   移动
def move():
    H = np.float32([[1, 0, 50], [0, 1, 25]])  # [1, 0, 50]表示X轴移动50，[0, 1, 25]表示Y轴移动25
    img2 = img1.copy()
    row, col = img2.shape[:2]  # 快速读取三维数组前俩个数组的长度
    print(img2.shape)
    print(row, col)
    print(H)
    new = cv2.warpAffine(img2, H, (col, row))  # 变换原始图像矩阵（移动）
    cv2.imshow('window1', img2)
    cv2.imshow('window2', new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#   缩放
def change_size():
    img3 = img1.copy()
    # 缩放有几种不同的插值（interpolation）方法，在缩小时推荐使用cv2.INTER_AREA，扩大时推荐使用cv2.INTER_CUBIC和cv2.INTER_LINEAR
    # 一是通过设置图像缩放比例，即缩放因子，来对图像进行放大或缩小
    res1 = cv2.resize(img3, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    height, width = img3.shape[:2]
    # 二是直接设置图像的大小，不需要缩放因子
    res2 = cv2.resize(img3, (int(0.8 * width), int(0.8 * height)), interpolation=cv2.INTER_AREA)
    cv2.imshow('origin_picture', img3)
    cv2.imshow('res1', res1)
    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#   旋转
def rotate():
    img = cv2.imread('4.jpg')
    rows, cols = img.shape[:2]
    # 第一个参数是旋转中心，第二个参数是旋转角度，第三个参数是缩放比例
    M1 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 0.5)
    M2 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 2)
    M3 = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    res1 = cv2.warpAffine(img, M1, (cols, rows))
    res2 = cv2.warpAffine(img, M2, (cols, rows))
    res3 = cv2.warpAffine(img, M3, (cols, rows))
    cv2.imshow('res1', res1)
    cv2.imshow('res2', res2)
    cv2.imshow('res3', res3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#   仿射
def affine():
    img = cv2.imread('4.jpg')
    rows, cols = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    # 类似于构造矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    res = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('原图', img)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#   透射
def transmission():
    img = cv2.imread('4.jpg')
    rows, cols = img.shape[:2]
    pts1 = np.float32([[56, 65], [238, 52], [28, 237], [239, 240]])
    pts2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    res = cv2.warpPerspective(img, M, (cols, rows))
    cv2.imshow('yuantu', img)
    cv2.imshow('res', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# img = salt(img1, 1000)
# img = white(img1, 1080)
# cv2.namedWindow('window1')
# cv2.imshow('window1', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# conduit(img1)
# drew()
# move()
# change_size()