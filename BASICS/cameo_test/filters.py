import cv2
import numpy
import BASICS.cameo_test.utils

def strokeEdges(src, dst, blurKsize = 7, edgeKsize = 5):
    # 此处若遇到性能问题，可以尝试减小blurKsize值，要关闭模糊效果，可将值设为3以下
    if blurKsize >=3:
        blurredSrc = cv2.medianBlur(src, blurKsize)  # 模糊处理
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)  # 边缘检测
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)  # 归一化，并乘以原图像？？？？
    channels = cv2.split(src)   # 通道拆分
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)

def test():
    # img = numpy.zeros((200, 200), dtype= numpy.uint8)   # 生成一个200x200的黑色区域
    # img[50: 150, 50: 150] = 255     # 将区域50x150变成白色
    img = cv2.imread("0221.png", 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)   # 将图片转换成二值化图像
    image, contours, hierarchay = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
    cv2.imshow('countours', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def test2():
    img = cv2.pyrUp(cv2.imread('0221.png', cv2.IMREAD_UNCHANGED))
    '''
    cv2.pyrDown：对图像进行高斯平滑，然后再降采样（将图像尺寸行和列方向缩减一半），
    若不指定第三个参数，则默认按照 Size((src.cols+1)/2, (src.rows+1)/2)计算
    
    cv2.pyrUp：对图像进行升采样（将图像尺寸行和列方向增大一倍），然后再进行高斯平滑，
    若不指定第三个参数，则默认按照  Size(src.cols*2, (src.rows*2)计算
    '''
    # 转换成灰度图后进行二值化处理
    ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # 查找轮廓
    '''
    RETR_LIST 从解释的角度来看，这中应是最简单的。它只是提取所有的轮廓，而不去创建任何父子关系。 
    RETR_EXTERNAL 如果你选择这种模式的话，只会返回最外边的的轮廓，所有的子轮廓都会被忽略掉。 
    RETR_CCOMP 在这种模式下会返回所有的轮廓并将轮廓分为两级组织结构。 
    RETR_TREE 这种模式下会返回所有轮廓，并且创建一个完整的组织结构列表。
    cv2.CHAIN_APPROX_NONE 表示边界所有点都会被储存
    cv2.CHAIN_APPROX_SIMPLE 会压缩轮廓，将轮廓上冗余点去掉
    '''
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)    # 计算边界框
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 画矩形

        # 转换最小轮廓的坐标后再绘图
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

        (x, y), radius = cv2.minEnclosingCircle(c)
        '''minEnclosingCircle返回值为：（圆心坐标， 圆半径）'''
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(img, center, radius, (0, 255, 0), 1)   # 画圆

    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)  # 绘制轮廓
    cv2.imshow('countours', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def epsilon(cnt):
    ep = 0.01 * cv2.arcLength(cnt, True)    # 计算轮廓长度
    '''
    cv2.approxPolyDP()计算近似的多边形框，
    第一个参数为“轮廓”;
    第二个参数为epsilon值，表示源轮廓与近似多边形的最大差值
    第三个参数为“布尔标记”，表示这个多边形是否闭合
    '''
    approx = cv2.approxPolyDP(cnt, ep, True)
    hull = cv2.convexHull(cnt)  # 计算凸包


def hough_lines():
    img = cv2.imread('F:\GIT\AITEST\BASICS\gray_29.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    minLineLength = 20
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges, 1, numpy.pi/180, 100, minLineLength, maxLineGap)
    '''
    HoughLines与HoughLinesP区别：
    HoughLines使用标准的Hough变换;
    HoughLinesP使用概率Hough变换，HoughLines的优化版，执行速度更快;
    接收一个由Canny边缘检测滤波器处理过的单通道二值化图像
    参数：
    一、表示需要处理的对象
    二、线段的几何表示th0/theta,一般分别取1和np.pi/180
    三、阈值，低于该阈值的线会被忽略
    四、最小直线长（更短的直线会被消除）和最大线段间隙（一条线段的间隙长度大于这个值会被视为俩条分开的线段）
    '''
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('edges', edges)
    cv2.imshow('lines', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def hough_circles():
    planets = cv2.imread('F:\GIT\AITEST\BASICS\hough_circle.png')
    gray_img = cv2.cvtColor(planets, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)
    # img2 = cv2.Canny(img.copy(), 50, 120)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300, param1=100, param2=30, minRadius=0, maxRadius=0)
    '''
    HoughCircles()参数：
    image:8位单通道图像（灰度图）
    method:检测方法
    dp:累加器分辨率与图像分辨率的反比，dp值越大，累加器数组越小
    minDist:检测到的圆中心之间的最小距离
    param1:用于处理边缘检测的梯度值方法
    param2:cv2.HOUGH_GRADIENT方法的累加器阈值。阈值越小，检测到的圈子越多
    minRadius：半径的最小大小（以像素为单位)
    maxRadius：半径的最大大小（以像素为单位）
    '''
    circles = numpy.uint16(numpy.around(circles))

    for i in circles[0,:]:
        cv2.circle(planets, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(planets, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imwrite('planets_circles.jpg', planets)
    cv2.imshow('houghcircles', planets)
    cv2.waitKey()
    cv2.destroyAllWindows()



hough_circles()