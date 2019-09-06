# encoding: utf-8
import cv2
import numpy as np
import os
import glob

# img = cv2.imread("1.jpg", 0)  # Canny只能处理灰度图，所以将读取的图像转成灰度图
#
# img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯平滑处理原图像降噪
# canny = cv2.Canny(img, 50, 150)  # apertureSize默认为3
#
# cv2.imshow('Canny', canny)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#   线上阈值处理
def CannyThreshold(lowThreshold):
    ratio = 3
    kernel_size = 3
    img = cv2.imread('1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # Canny只能处理灰度图，所以将读取的图像转成灰度图
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo', dst)

def Threshold():
    lowThreshold = 0
    max_lowThreshold = 100
    cv2.namedWindow('canny demo')
    cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold, max_lowThreshold, CannyThreshold)
    CannyThreshold(0)  # initialization
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()

#   录制并保存视频
def save_redio():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('camera_test2.avi', fourcc, 10.0, size)
    while True:
        ret, frame = cap.read()
        # 横向翻转
        frame = cv2.flip(frame, 1)
        out.write(frame)
        # 在图像上显示 Press Q to save and quit
        cv2.putText(frame,
                    "Press Q to save and quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

#   读取视频
def read_redio():
    video_path = r'F:\GIT\AITEST\BASICS'
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        try:
            base = video_name.split('.')[1]
        except Exception as e:
            print(e)
            break
        if base == u'avi':
            folder_name = file_name
            vc = cv2.VideoCapture(video_path + u'\\' + video_name)  # 读入视频文件
            c = 1
            if vc.isOpened():  # 判断是否正常打开
                rval, frame = vc.read()
                os.makedirs(folder_name, exist_ok=True)
            else:
                rval = False

            timeF = 300  # 视频帧计数间隔频率

            while rval:  # 循环读取视频帧
                rval, frame = vc.read()
                pic_path = folder_name + '/'
                if (c % timeF == 0):  # 每隔timeF帧进行存储操作
                    cv2.imwrite(pic_path + file_name + '_' + str(c) + '.jpg', frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
                c = c + 1
                cv2.waitKey(1)
            vc.release()
        else:
            print(video_name)

#   边录制边读取视频
def sys_redio():
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # 判断是否正常打开
        print('camera is open')
    else:
        print('camera is not open')
    '''
    FourCC全称Four-Character Codes，代表四字符代码 (four character code), 
    它是一个32位的标示符，其实就是typedef unsigned int FOURCC;是一种独立标示视频数据流格式的四字符代码。
    因此cv2.VideoWriter_fourcc()函数的作用是输入四个字符代码即可得到对应的视频编码器
    (*'XVID')表示使用XVID编码器
    '''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('camera_test3.avi', fourcc, 10.0, size)   # 参数分别为：视频名，编码器，帧率（正常是10.0），视频宽高
    print('size:' + repr(size))


    kernel = np.ones((5, 5), np.uint8)
    background = None

    while True:
        # 读取视频流
        grabbed, frame_lwpCV = vc.read()
        # 对帧进行预处理，先转灰度图，再进行高斯滤波。
        # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
        gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
        gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)

        # 将第一帧设置为整个输入的背景
        if background is None:
            background = gray_lwpCV
            continue
        # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
        # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
        diff = cv2.absdiff(background, gray_lwpCV)
        diff = cv2.threshold(diff, 148, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
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
        image, contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
        for c in contours:
            if cv2.contourArea(c) < 6000:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
                continue
            (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框，元素c是一个二值图，x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
            cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 根据boundingRect计算结果画出矩形，此时图片参数为原图
            tip_w = str(w)
            tip_h = str(h)
            tip_x = int(x+w/2)
            # tip_y = int(y-h)
            tip_y = int(y)
            cv2.putText(frame_lwpCV, tip_w, (tip_x, tip_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        cv2.imshow('contours', frame_lwpCV)
        # cv2.imshow('dis', diff)

        key = cv2.waitKey(1) & 0xFF
        # 按'q'健退出循环
        if key == ord('q'):
            break
    # When everything done, release the capture
    vc.release()
    cv2.destroyAllWindows()


#   拍照
def get_photo1():
    cap = cv2.VideoCapture(0)   # 0表示第一个摄像头
    ret, frame = cap.read()
    cv2.imwrite('photo.jpeg', frame)
    cap.release()
    cv2.destroyAllWindows()

def get_photo2():
    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        cv2.imshow('photo window', frame)
        '''
        若cv2.waitKey(n),n为0，则不会有返回值，也就是会一直停留在摄像头，若n>0，则会等待n毫秒之后，若无按键，则返回-1，有按键则返回键盘值
        0xFF表示十六进制常数，通过使用位运算符&，表示只留下后八位，目的是防止键盘输入的值过大，则利用该运算符只取字符最后一个字节
        ord()返回对应的ASCII数值
        '''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('photo2.jpeg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys_redio()
    # get_photo2()
