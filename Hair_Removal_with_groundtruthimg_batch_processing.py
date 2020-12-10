# 本程序批量处理指定文件夹下的所有待处理图像。
# 更改dataset_path为待处理图像文件夹的路径，gdtruthset_path为对应的groundTruth图像文件夹路径，output_path为处理结果图像输出路径
# width和height分别为resize后的图像宽高，可以自定设定，默认不进行resize
# 注意：resize可能会使得图像处理的质量发生显著改变，若不想resize只需把width和height赋值为0

from imutils import paths
import cv2 as cv
import numpy as np


# 获取所有待处理图片的路径
dataset_path='data/raw'  # 存放着待处理图片的文件夹路径
imagePaths = sorted(list(paths.list_images(dataset_path)))
gdtruthset_path='data/groundTruth'  # 存放着groundTruth图像的文件夹路径
gdtruthPaths = sorted(list(paths.list_images(gdtruthset_path)))
output_path = 'data/output/'  # 处理结果输出路径

# resize图像的尺寸
width = 1080
height = 720


def getHairMask(img, gdt):
	"""
	根据经过通道处理后的图像返回hair mask
	:param img: 经过通道处理后的图像
	:param gdt: groundTruth图像
	:return: hair mask
	"""
	img1 = img
	# OTSU阈值分割，得到2值画图像
	t, img2 = cv.threshold(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

	kr = cv.getStructuringElement(cv.MORPH_ELLIPSE, (40, 40))  # kernel，长轴和短轴均为40的椭圆，即半径20的圆
	img3 = cv.morphologyEx(img2, cv.MORPH_TOPHAT, kr)  # 礼帽/顶帽运算
	kr1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
	img5 = cv.dilate(img3, kr1)  # 膨胀

	# 使用ground truth来保持病灶边缘的清晰
	gdt = cv.cvtColor(gdt, cv.COLOR_BGR2GRAY)  # 转换为灰度图
	t1, gdt = cv.threshold(gdt, 128, 255, cv.THRESH_BINARY_INV)  # 转换为反向2值图像
	img6 = cv.bitwise_and(img5, gdt)  # 进行逻辑与运算，得到hair mask
	return img6


for imgpath, gdtruthpath in zip(imagePaths, gdtruthPaths):
	# 读取图像，resize
	origin_img=cv.imread(imgpath)
	gdtruth = cv.imread(gdtruthpath)
	if width!=0 and height!=0:
		origin_img=cv.resize(origin_img, (width, height))
		gdtruth = cv.resize(gdtruth, (width, height))

	# 分通道处理
	# ------------------------------------------------------------------------------------------------------
	b, g, r = cv.split(origin_img)

	# 对各个通道进行中值滤波
	b = cv.medianBlur(b, 3)
	g = cv.medianBlur(g, 3)
	r = cv.medianBlur(r, 3)

	# 合并处理后的通道，重新得到BGR图像
	img1 = cv.merge([b, g, r])

	# create hair mask
	# ------------------------------------------------------------------------------------------------------
	hairmask = getHairMask(img1, gdtruth)

	# 去除毛发
	# ------------------------------------------------------------------------------------------------------
	result_img = cv.inpaint(img1, hairmask, 3, flags=cv.INPAINT_TELEA)

	# 输出处理结果和原图的对比图片，存储到本地
	# ------------------------------------------------------------------------------------------------------
	compare_img = np.concatenate((origin_img, result_img), 1)
	fname=output_path+imgpath.split('\\')[1]  # 输出图片的存储路径
	cv.imwrite(fname,compare_img)
