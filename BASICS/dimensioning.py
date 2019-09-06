# encoding: utf-8
import cv2
import numpy as np
import os

KNOWN_DISTANCE = 35.0   # 相机对白纸的距离
KNOWN_WIDTH = 7.0  # 白纸已知宽度

#   返回轮廓数组
# def find_marker(image):
#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 获得单通道灰度图
#     gray = cv2.GaussianBlur(gray, (5, 5), 0)    # 高斯模糊去噪（参数必须为单通道灰度图）
#     ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 获得二值化图像
#     cv2.Canny(gray, 35, 125)    # 进行边缘检测
#
#     binary, cnts, _ = cv2.findContours(binary, cv2.RETR_LIST,
#                                        cv2.CHAIN_APPROX_NONE)  # 检测物体轮廓
#
#     c = max(cnts, key=cv2.contourArea)  # 以key为判断条件求cnts(轮廓)的最大值
#     cv2.drawContours(image, c, -1, (0, 255, 0), 3)  # 绘制该最大轮廓
#     # compute the bounding box of the of the paper region and return it
#     # 返回包含 (x, y) 坐标和像素高度和宽度信息的边界框给调用函数
#     # inAreaRect返回矩形的中心点坐标，长宽，旋转角度
#     # 角度值取值范围为[-90,0)，当矩形水平或竖直时均返回-90
#
#     array = cv2.minAreaRect(c)
#     return array

#   缩放图片
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]    	# 初始化缩放比例，并获取图像尺寸

    # 如果宽度和高度均为0，则返回原图
    if width is None and height is None:
        return image

    # 宽度是0
    if width is None:
        # 则根据高度计算缩放比例
        r = height / float(h)
        dim = (int(w * r), height)

    # 如果高度为0
    else:
        # 根据宽度计算缩放比例
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def actual_width(knownDistance, focalLength, focalWidth):
    return (knownDistance * focalWidth / focalLength)


#   计算焦距
def caculate_focal_length():
    distance = 17   # cm
    object_width = 19  # cm
    img = cv2.imread('photo2.jpeg')
    gray_lwpCV = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
    ret, binary = cv2.threshold(gray_lwpCV, 127, 255, cv2.THRESH_BINARY)
    binary, cnts, _ = cv2.findContours(binary, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)  # RETR_TREE   cv2.CHAIN_APPROX_SIMPLE)

    c = max(cnts, key=cv2.contourArea)
    print(c)
    img2 = img.copy()
    # cv2.drawContours(img2, c, -1, (0, 255, 0), 3)
    # array = cv2.minAreaRect(c)
    (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框，元素c是一个二值图，x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 根据boundingRect计算结果画出矩形，此时图片参数为原图
    # cv2.imshow('photo3', img2)
    cv2.imwrite('photo4.jpeg', img2)
    pixel_width = w
    focal_length = distance * pixel_width / object_width
    print(x, y, w, h)
    print('focal_length:', focal_length)

    return focal_length


#   边录制边读取视频
def sys_redio():
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # 判断是否正常打开
        print('camera is open')
    else:
        print('camera is not open')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cv2.VideoWriter('camera_test3.avi', fourcc, 10.0, size)   # 参数分别为：视频名，编码器，帧率（正常是10.0），视频宽高
    print('size:' + repr(size))

    # kernel = np.ones((5, 5), np.uint8)
    background = None

    while True:
        # 读取视频流
        grabbed, frame_lwpCV = vc.read()
        # 对帧进行预处理，先转灰度图，再进行高斯滤波。
        # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
        gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)

        # step2:用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
        gradX = cv2.Sobel(gray_lwpCV, cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray_lwpCV, cv2.CV_32F, dx=0, dy=1, ksize=-1)

        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # show image
        cv2.imshow("first", gradient)
        cv2.waitKey()

        gray_lwpCV = cv2.GaussianBlur(gradient, (21, 21), 0)
        # gray_lwpCV = cv2.blur(gradient, (9, 9))

        # show image
        # cv2.imshow("camer2", gray_lwpCV)
        # cv2.imshow("camer3", gray_lwpCV2)
        #
        # cv2.waitKey()

        # 将第一帧设置为整个输入的背景
        if background is None:
            background = gray_lwpCV
            continue
        # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
        # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
        diff = cv2.absdiff(background, gray_lwpCV)
        diff = cv2.threshold(diff, 90, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
        # show image
        # cv2.imshow("camer4", diff)
        # cv2.waitKey()
        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
        diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀
        # 显示矩形框
        '''
        findContour()函数用于查找物理轮廓，其条件是传入图片必须是二值化，普通图片需要转化为灰度图再转化为二值化
        cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])  
        第一个参数是寻找轮廓的图像；
        第二个参数表示轮廓的检索模式，有四种（本文介绍的都是新的cv2接口）：
        cv2.RETR_EXTERNAL表示只检测外轮廓
        cv2.RETR_LIST检测的轮廓不建立等级关系
        cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        cv2.RETR_TREE建立一个等级树结构的轮廓。
        第三个参数method为轮廓的近似办法
        cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
        '''
        image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_LIST,
                                                      cv2.CHAIN_APPROX_NONE)  # 该函数计算一幅图像中目标的轮廓
        con = []
        for i in contours:
            con.append(cv2.contourArea(i))

        for i in contours:
            if len(con) == 0:
                continue
            elif cv2.contourArea(i) == max(con):
                (x, y, w, h) = cv2.boundingRect(i)  # 该函数计算矩形的边界框，元素c是一个二值图，x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
                cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 根据boundingRect计算结果画出矩形，此时图片参数为原图

                focalLength = caculate_focal_length()
                ac_width = size_caculate(w, 17, focalLength)
                tip_w = str(w)
                tip_h = str(h)
                tip_x = int(x+w/2)
                tip_y = int(y-h)
                cv2.putText(frame_lwpCV, str(ac_width), (tip_x, tip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        cv2.imshow('contours', frame_lwpCV)
        # cv2.imshow('dis', diff)

        key = cv2.waitKey(1) & 0xFF
        # 按'q'健退出循环
        if key == ord('q'):
            break
    # When everything done, release the capture
    vc.release()
    cv2.destroyAllWindows()

#   计算实际宽度
def size_caculate(pixel_width,distance, focalLength):

    return pixel_width * distance / focalLength

if __name__ == '__main__':

    sys_redio()
    # caculate_focal_length()