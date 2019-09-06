import cv2
import numpy as np
from matplotlib import pyplot as plt
# import time
import random

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
def move(self):
    H = np.float32([[1, 0, 50], [0, 1, 25]])  # [1, 0, 50]表示X轴移动50，[0, 1, 25]表示Y轴移动25
    img2 = self.img1.copy()
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

class ShapeAnalysis:

    def __init__(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}

    def analysis(self, frame):
        h, w, ch = frame.shape
        result = np.zeros((h, w, ch), dtype=np.uint8)
        # 二值化图像
        print("start to detect lines...\n")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("input image", frame)
        out_binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in range(len(contours)):

            # 计算面积与周长
            p = cv2.arcLength(contours[cnt], True)
            area = cv2.contourArea(contours[cnt])

            if area > 10000:
                # 提取与绘制轮廓
                cv2.drawContours(result, contours, cnt, (0, 255, 0), 2)

                # 轮廓逼近
                epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
                approx = cv2.approxPolyDP(contours[cnt], epsilon, True)

                # 分析几何形状
                corners = len(approx)
                shape_type = ""
                if corners == 3:
                    count = self.shapes['triangle']
                    count = count + 1
                    self.shapes['triangle'] = count
                    shape_type = "三角形"
                if corners == 4:
                    count = self.shapes['rectangle']
                    count = count + 1
                    self.shapes['rectangle'] = count
                    shape_type = "矩形"
                if corners >= 10:
                    count = self.shapes['circles']
                    count = count + 1
                    self.shapes['circles'] = count
                    shape_type = "圆形"
                if 4 < corners < 10:
                    count = self.shapes['polygons']
                    count = count + 1
                    self.shapes['polygons'] = count
                    shape_type = "多边形"

                # 求解中心位置
                mm = cv2.moments(contours[cnt])
                if mm['m10'] or mm['m00'] or mm['m01'] == 0:
                    cx = 0
                    cy = 0
                else:
                    cx = int(mm['m10'] / mm['m00'])
                    cy = int(mm['m01'] / mm['m00'])
                cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)

                # 颜色分析
                color = result[cy][cx]
                color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"

                print("周长: %.3f, 面积: %.3f 颜色: %s 形状: %s " % (p, area, color_str, shape_type))


        # cv2.imshow("Analysis Result", self.draw_text_info(result))
        # cv2.imwrite("D:/test-result.png", self.draw_text_info(result))
        return self.shapes

    def draw_text_info(self, image):
        c1 = self.shapes['triangle']
        c2 = self.shapes['rectangle']
        c3 = self.shapes['polygons']
        c4 = self.shapes['circles']
        cv2.putText(image, "triangle: " + str(c1), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "rectangle: " + str(c2), (10, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "polygons: " + str(c3), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        cv2.putText(image, "circles: " + str(c4), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 1)
        return image

class VideoAnalysis:

    def __init__(self):
        self.shapes = {'triangle': 0, 'rectangle': 0, 'polygons': 0, 'circles': 0}


    def analysis(self, frame):
        # h, w, ch = frame.shape
        # result = np.zeros((h, w, ch), dtype=np.uint8)
        # 二值化图像
        print("start to detect lines...\n")
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # cv2.imshow("input image", frame)
        con = []
        out_binary, contours, hierarchy = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in range(len(contours)):

            # 计算面积与周长
            p = cv2.arcLength(contours[cnt], True)
            area = cv2.contourArea(contours[cnt])
            print(area)
            if area > 5000:
                # 提取与绘制轮廓
                # cv2.drawContours(frame, contours, cnt, (0, 255, 0), 2)

                # 轮廓逼近
                epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
                approx = cv2.approxPolyDP(contours[cnt], epsilon, True)

                # 分析几何形状
                corners = len(approx)
                shape_type = ""
                if corners == 3:
                    count = self.shapes['triangle']
                    count = count + 1
                    self.shapes['triangle'] = count
                    shape_type = "三角形"
                    # con.append(contours[cnt])


                if corners == 4:
                    count = self.shapes['rectangle']
                    count = count + 1
                    self.shapes['rectangle'] = count
                    shape_type = "矩形"
                    con.append(contours[cnt])

                if corners >= 10:
                    count = self.shapes['circles']
                    count = count + 1
                    self.shapes['circles'] = count
                    shape_type = "圆形"
                    con.append(contours[cnt])


                if 4 < corners < 10:
                    count = self.shapes['polygons']
                    count = count + 1
                    self.shapes['polygons'] = count
                    shape_type = "多边形"
                    con.append(contours[cnt])


                # # 求解中心位置
                # mm = cv2.moments(contours[cnt])
                # if mm['m10'] or mm['m00'] or mm['m01'] == 0:
                #     cx = 0
                #     cy = 0
                # else:
                #     cx = int(mm['m10'] / mm['m00'])
                #     cy = int(mm['m01'] / mm['m00'])
                # cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                #
                # # 颜色分析
                # color = frame[cy][cx]
                # color_str = "(" + str(color[0]) + ", " + str(color[1]) + ", " + str(color[2]) + ")"
                #
                # print("周长: %.3f, 面积: %.3f 颜色: %s 形状: %s " % (p, area, color_str, shape_type))


        # cv2.imshow("Analysis Result", self.draw_text_info(result))
        # cv2.imwrite("D:/test-result.png", self.draw_text_info(result))
        return con

    #   边录制边读取视频
    def sys_redio(self):
        vc = cv2.VideoCapture(0)
        if vc.isOpened():  # 判断是否正常打开
            print('camera is open')
        else:
            print('camera is not open')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cv2.VideoWriter('camera_test3.avi', fourcc, 10.0, size)  # 参数分别为：视频名，编码器，帧率（正常是10.0），视频宽高
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
            gray_lwpCV = cv2.blur(gradient, (9, 9))

            # gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
            focalLength = self.caculate_focal_length()
            # 将第一帧设置为整个输入的背景
            if background is None:
                background = gray_lwpCV
                continue
            # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
            # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
            diff = cv2.absdiff(background, gray_lwpCV)
            # cv2.imshow('diff', diff)
            # cv2.waitKey(0)
            diff = cv2.threshold(diff, 90, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
            es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
            diff = cv2.dilate(diff, es, iterations=4)  # 形态学膨胀
            kernel = np.ones((5, 5), np.uint8)
            diff = cv2.erode(diff, kernel, iterations=4)
            # cv2.imshow('diff', diff)
            # cv2.waitKey(0)
            countour = self.analysis(diff)
            # 显示矩形框
            for i in countour:
                if len(countour) == 0:
                    continue
                # elif cv2.contourArea(i) == max(self.con):
                else:
                    cv2.drawContours(frame_lwpCV, i, -1, (0,255,0), 3)
                    # (x, y, w, h) = cv2.boundingRect(i)  # 该函数计算矩形的边界框，元素c是一个二值图，x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
                    # cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0),
                    #               2)  # 根据boundingRect计算结果画出矩形，此时图片参数为原图

                    # ac_width = self.size_caculate(w, 17, focalLength)
                    # tip_w = str(w)
                    # tip_h = str(h)
                    # tip_x = int(x + w / 2)
                    # tip_y = int(y - h)
                    # cv2.putText(frame_lwpCV, str(ac_width), (tip_x, tip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    #             (100, 100, 100), 2)
            countour = []
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
    def size_caculate(self, pixel_width, distance, focalLength):

        return pixel_width * distance / focalLength

    #   计算焦距
    def caculate_focal_length(self):
        distance = 17  # cm
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

    def test(self):
        # step1：加载图片，转成灰度图
        image = cv2.imread("test_2803.jpg")
        hsv_lwpCV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_lwpCV = cv2.blur(hsv_lwpCV, (9, 9))
        lower_red = np.array([30, 12, 55])
        upper_red = np.array([50, 255, 255])

        hsv_lwpCV = cv2.inRange(hsv_lwpCV, lower_red, upper_red)
        # print(hsv_lwpCV)
        # cv2.waitKey(0)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (255, 255))
        # closed = cv2.morphologyEx(hsv_lwpCV, cv2.MORPH_CLOSE, kernel)
        #
        # closed = cv2.erode(closed, None, iterations=4)
        # closed = cv2.dilate(closed, None, iterations=4)
        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
        kernel = np.ones((5, 5), np.uint8)
        diff = cv2.dilate(hsv_lwpCV, es, iterations=4)  # 形态学膨胀
        diff = cv2.erode(diff, kernel, iterations=4)
        cv2.imshow('hsv_lwpCV', hsv_lwpCV,)
        cv2.imshow('closed', diff)
        cv2.waitKey(0)
        img, contours, hier = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # _a, cnts, _b = x
        # c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # OpenCV中通过cv2.drawContours在图像上绘制轮廓。
        # 第一个参数是指明在哪幅图像上绘制轮廓
        # 第二个参数是轮廓本身，在Python中是一个list
        # 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓
        # 第四个参数是轮廓线条的颜色
        # 第五个参数是轮廓线条的粗细

        # cv2.minAreaRect()函数:
        # 主要求得包含点集最小面积的矩形，这个矩形是可以有偏转角度的，可以与图像的边界不平行。
        # compute the rotated bounding box of the largest contour
        # rect = cv2.minAreaRect(c)
        # rect = cv2.minAreaRect(cnts[1])
        rect = max(contours, key=cv2.contourArea)
        print(rect)
        box =cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            print(x,y,w,h)
            cv2.rectangle(image, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
            # cv2.rectangle(image, (x, y), (x+w), (y+h), (0, 255, 0), 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image, [box], 0, (0, 0 ,255), 3)

        cv2.imshow('image', image)
        cv2.waitKey(0)



        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # step2:用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
        gradX = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=-1)
        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        # show image
        # cv2.imshow("first", gradient)
        # cv2.waitKey()

        # step3：去除图像上的噪声。首先使用低通滤泼器平滑图像（9 x 9内核）,这将有助于平滑图像中的高频噪声。
        # 低通滤波器的目标是降低图像的变化率。如将每个像素替换为该像素周围像素的均值。这样就可以平滑并替代那些强度变化明显的区域。
        # 然后，对模糊图像二值化。梯度图像中不大于90的任何像素都设置为0（黑色）。 否则，像素设置为255（白色）。
        # blur and threshold the image
        blurred = cv2.blur(gradient, (9, 9))
        _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY_INV)
        # SHOW IMAGE
        cv2.imshow("thresh", thresh)
        cv2.waitKey()

        # step4:在上图中我们看到蜜蜂身体区域有很多黑色的空余，我们要用白色填充这些空余，使得后面的程序更容易识别昆虫区域，
        # 这需要做一些形态学方面的操作。
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # show image
        cv2.imshow("closed1", closed)
        cv2.waitKey()

        # step5:从上图我们发现图像上还有一些小的白色斑点，这会干扰之后的昆虫轮廓的检测，要把它们去掉。分别执行4次形态学腐蚀与膨胀。
        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        # show image
        cv2.imshow("closed2", closed)
        cv2.waitKey()

        # step6：找出昆虫区域的轮廓。
        # cv2.findContours()函数
        # 第一个参数是要检索的图片，必须是为二值图，即黑白的（不是灰度图），
        # 所以读取的图像要先转成灰度的，再转成二值图，我们在第三步用cv2.threshold()函数已经得到了二值图。
        # 第二个参数表示轮廓的检索模式，有四种：
        # 1. cv2.RETR_EXTERNAL表示只检测外轮廓
        # 2. cv2.RETR_LIST检测的轮廓不建立等级关系
        # 3. cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        # 4. cv2.RETR_TREE建立一个等级树结构的轮廓。
        # 第三个参数为轮廓的近似方法
        # cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

        # cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。
        # cv2.findContours()函数返回第一个值是list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。
        # 每一个ndarray里保存的是轮廓上的各个点的坐标。我们把list排序，点最多的那个轮廓就是我们要找的昆虫的轮廓。
        x = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # import pdb
        # pdb.set_trace()
        _a, cnts, _b = x
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        # OpenCV中通过cv2.drawContours在图像上绘制轮廓。
        # 第一个参数是指明在哪幅图像上绘制轮廓
        # 第二个参数是轮廓本身，在Python中是一个list
        # 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓
        # 第四个参数是轮廓线条的颜色
        # 第五个参数是轮廓线条的粗细

        # cv2.minAreaRect()函数:
        # 主要求得包含点集最小面积的矩形，这个矩形是可以有偏转角度的，可以与图像的边界不平行。
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        # rect = cv2.minAreaRect(cnts[1])
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # draw a bounding box arounded the detected barcode and display the image
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        cv2.imshow("Image", image)
        cv2.imwrite("contoursImage2.jpg", image)
        cv2.waitKey(0)

        # step7：裁剪。box里保存的是绿色矩形区域四个顶点的坐标。我将按下图红色矩形所示裁剪昆虫图像。
        # 找出四个顶点的x，y坐标的最大最小值。新图像的高=maxY-minY，宽=maxX-minX。
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        cropImg = image[y1:y1 + hight, x1:x1 + width]

        # show image
        cv2.imshow("cropImg", cropImg)
        cv2.imwrite("bee.jpg", cropImg)
        cv2.waitKey()

class ImageExtraction:

    def test1(self):
        img = cv2.imread('test_2803.jpg')
        mask = np.zeros(img.shape[:2], np.uint8)    # 建掩模
        print(img.shape)
        bgdModel = np.zeros((1, 65), np.float64)    # 建前景图
        fgbModel = np.zeros((1, 65), np.float64)    # 建背景图
        rect = (100, 50, 1080, 1080)    # 选择处理范围
        cv2.grabCut(img, mask, rect, bgdModel, fgbModel, 5, cv2.GC_INIT_WITH_RECT)  # 图片分割
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')   # 像素转换，像素为2/0都转为0,1转为1
        img = img * mask2[:,:,np.newaxis]
        print(img)
        plt.subplot(121), plt.imshow(img)
        plt.title("grabcut"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread('test_2803.jpg'), cv2.COLOR_BGR2RGB))
        plt.title("original"), plt.xticks([]), plt.yticks([])
        plt.show()

    # 分水岭算法
    def test2(self):
        img = cv2.imread('test_2803.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用自适应阈值分割，高斯邻域
        img_2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        cv2.blur(img_2, (9,9))
        # cv2.imshow('img_2', img_2)
        # cv2.waitKey(0)
        # 去除噪声，阈值90以下的全部转换成白色
        ret, thresh = cv2.threshold(img_2, 90, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        cv2.waitKey(0)
        # 使用morphologyEx膨胀后再腐蚀
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)     # 前景
        cv2.imshow('sure_bg', sure_bg)
        cv2.waitKey(0)

        # 距离变换函数，目的是目标细化、骨架提取、形状差值及匹配、黏连物体分离等，再利用阈值得到最大的前景
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)  # 背景
        cv2.imshow('dist_transform', dist_transform)
        cv2.imshow('sure_fg', sure_fg)
        cv2.waitKey(0)

        # 去掉前景与背景重合的部分
        sure_fg = np.uint8(sure_fg)
        unknow = cv2.subtract(sure_bg, sure_fg)
        cv2.imshow('unknow', unknow)
        cv2.waitKey(0)
        # 设定栅栏
        ret, markers = cv2.connectedComponents(unknow)
        markers = markers + 1
        markers[unknow == 255] = 0
        # 打开栅栏
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]
        plt.imshow(img)
        plt.show()

class faceAnalysis:

    def detect(self, frame):

        face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')  # 初始化haar特征分类器
        # img = cv2.imread(filename)
        eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        '''
        deteceMultiScale参数：
        image: 输入图像，单通道灰度图
        scaleFactor：每次缩小图像的比例，默认1.1
        minNeighbors：匹配成功所需要的周围矩形框的数据，每个特征匹配到的区域都是一个矩形框，只有多个矩形框
        同时存在的时候才认为是匹配成功，默认3
        flags：可取值
            CASCADE_DO_CANNY_PRUNING=1, 利用canny边缘检测来排除一些边缘很少或者很多的图像区域
            CASCADE_SCALE_IMAGE=2, 正常比例检测
            CASCADE_FIND_BIGGEST_OBJECT=4, 只检测最大的物体
            CASCADE_DO_ROUGH_SEARCH=8 初略的检测
         minSize=None, maxSize=None：匹配物体的大小范围
        '''
        all_tip = []
        eyes_tip = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]   # 在面部区域识别眼睛
            eye = eye_cascade.detectMultiScale(roi_gray, 1.1, 3, 0, (40, 40))
            for (ex, ey, ew, eh) in eye:
                eyes_tip.append((ex, ey, ew, eh))
            dict = {'face_tip': (x, y, w, h), 'eyes_tip': eyes_tip}
            all_tip.append(dict)
            eyes_tip = []
        return all_tip

    def video_face_analysis(self):
        vc = cv2.VideoCapture(0)
        if vc.isOpened():  # 判断是否正常打开
            print('camera is open')
        else:
            print('camera is not open')

        while True:
            # 读取视频流
            grabbed, frame_lwpCV = vc.read()
            all_tips = self.detect(frame_lwpCV)
            for all_tip in all_tips:
                (x, y, w, h) = all_tip['face_tip']
                cv2.rectangle(frame_lwpCV, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)
                for (ex, ey, ew, eh) in all_tip['eyes_tip']:
                    roi_frame = frame_lwpCV[y:y + h, x:x + w]
                    cv2.rectangle(roi_frame, (ex, ey), ((ex + ew), (ey + eh)), (255, 0, 0), 2)
            cv2.imshow('contours', frame_lwpCV)

            key = cv2.waitKey(1) & 0xFF
            # 按'q'健退出循环
            if key == ord('q'):
                break
        # When everything done, release the capture
        vc.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # src = cv2.imread("test_2601.png")
    # ld = ShapeAnalysis()
    # ld.analysis(src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Id = VideoAnalysis()
    # Id.sys_redio()
    # Id.test()
    # image_extraction = ImageExtraction()
    # image_extraction.test2()
    face_analysis = faceAnalysis()
    # face_analysis.detect()
    face_analysis.video_face_analysis()
    # time = 100000
    # count = 0
    # for i in range(time):
    #     box = [0,0,0]
    #     my_choice = random.randint(1,3)
    #     box[random.randint(1,3)-1] = 1
    #     if (i)%30 == 0:
    #         print("\n")
    #     if box[my_choice - 1]:
    #         count = count + 1
    #         print("■",end=" ")
    #     else:
    #         print("□",end=" ")
    # print("\n",count*100/time,"%")



# contour_test()