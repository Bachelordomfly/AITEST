import cv2
import os
from scipy import ndimage
import numpy as np
from matplotlib import pyplot as plt

def test1():
    cameraCapture = cv2.VideoCapture(0)
    fps = 30    # 对帧率做出假设，目的是针对摄像头创建合适的VideoWriter类
    size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('MyOutputVid.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    success, frame = cameraCapture.read()   # success判断摄像头是否正确打开， frame表示摄像头读到的数据

    numFrameRemaining = 10 * fps - 1
    while success and numFrameRemaining > 0:
        videoWriter.write(frame)
        success, frame = cameraCapture.read()
        numFrameRemaining -= 1

    # 多组摄像头或3D摄像头使用grab()、retrieve()代替read()
    # success1 = cameraCapture0.grab()
    # success2 = cameraCapture1.grab()
    # if success1 and success2:
    #     frame0 = cameraCapture0.retrieve()
    #     frame1 = cameraCapture1.retrieve()
    cameraCapture.release()

    '''
        1、定义一个视频程序，0表示打开笔记本内置摄像头，参数若是视频路径则是打开视频，如cv2.VideoCapture("../test.avi")
        2、size=(视频的像素宽， 视频的像素高)
        3、VideoWriter(const string& filename, int fourcc, double fps, Size frameSize, bool isColor=true)
        fourcc表示使用的编码方式，若输入-1，则弹出提示框选择
        fourcc为 四个字符用来表示压缩帧的codec 例如：
        CV_FOURCC('P','I','M','1') = MPEG-1 codec
        CV_FOURCC('M','J','P','G') = motion-jpeg codec
        CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
        CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
        CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
        CV_FOURCC('U', '2', '6', '3') = H263 codec
        CV_FOURCC('I', '2', '6', '3') = H263I codec
        CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec
        4、fps表示创建视频流的帧率，正常帧率为30，<30会感受到卡顿
        5、3D立体摄像头或kinect主要指标为：物距、间距、夹角
        
    '''

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

def test4():
    cameraCapture = cv2.VideoCapture(0)
    cv2.namedWindow('MyWindow')
    cv2.setMouseCallback('MyWindow', onMouse)   # 获取鼠标输入

    print('show camera feed. click window or press any key to stop.')
    success, frame = cameraCapture.read()
    '''
    waitKey()的返回值为-1或ASCII码，但是有可能比ASCII码更大，以下面的方式来读取返回值的最后一个字节来保证只提取ASCII码
    keycode = cv2.waitKey(1)
    if keycode != -1:
        keycode &= 0xFF
        
    参数0表示只显示当前帧图像，1表示延时1ms切换到下一帧图像
    '''
    while success and cv2.waitKey(1) == -1 and not clicked:
        cv2.imshow('MyWindow', frame)
        success, frame = cameraCapture.read()
    cv2.destroyAllWindows()
    cameraCapture.release()
# test1()

# def test2():
    # img = np.zeros((3,3), dtype=np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print(img)
    # print(img.shape)
    # grayImg = cv2.imread('gray_29.jpg', cv2.IMREAD_GRAYSCALE)
    # print(grayImg.shape)
    # print(grayImg)
    # cv2.imwrite('gray_30.png', grayImg)
    # randomByteArray = bytearray(os.urandom(120000))
    # flatNumpyArray = np.array(randomByteArray)
    # grayImage = flatNumpyArray.reshape(300, 400)
    # np.random.randint(0, 256, 120000).reshape(300, 400)

    # cv2.imwrite('randowImg.png', grayImage)
    # bgrImg = flatNumpyArray.reshape(100, 400, 3)
    # cv2.imwrite('bgrImg.png', bgrImg)

    # img = np.zeros((200, 256, 3), dtype=np.uint8)
    # print(img)
    #
    # img[:, :, 1] = 175
    # print(img)
    # img[50:100, 50:100] = 256
    # cv2.imwrite('175.png', img)
    #
    # print(img.dtype)
    # print(img.size)
    # print(img.shape)

    # videoCapture = cv2.VideoCapture('camera_test2.avi')
    # fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # videoWrite = cv2.VideoWriter('videotest1.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    # success, frame = videoCapture.read()
    # while success:
    #     videoWrite.write(frame)
    #     success, frame = videoCapture.read()

    # cameraCapture = cv2.VideoCapture(0)
    # fps = 30
    # size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # videoWrite = cv2.VideoWriter('10time.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    # success, frame = cameraCapture.read()
    # numFramesRemaining = 10 * fps - 1
    # while success and numFramesRemaining > 0:
    #     videoWrite.write(frame)
    #     success, frame = cameraCapture.read()
    #     numFramesRemaining -= 1
    # cameraCapture.release()


def test5():
    kernel_3x3 = np.array([[-1, -1, -1], [-1, 0, -1], [-1, -1, -1]])
    kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, 2, 4, 2, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, -1, -1, -1, -1]])

    img = cv2.imread('gray_29.jpg', 0)
    k3 = ndimage.convolve(img, kernel_3x3)  # 仿高通滤波器
    k5 = ndimage.convolve(img, kernel_5x5)

    blurred = cv2.GaussianBlur(img, (11, 11), 0)    # 低通滤波器
    g_hpf = img - blurred
    cv2.imshow("3x3", k3)
    cv2.imshow("5x5", k5)
    cv2.imshow("g_hpf", g_hpf)
    cv2.waitKey()
    cv2.destroyAllWindows()

