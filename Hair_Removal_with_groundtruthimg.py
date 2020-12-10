# 本程序用于处理单张图片，请给image_path传递待处理的图片路径，给groundtruth_path传递待处理图片对应的groundTruth图像路径
# new_width和new_height分别为resize后的图像宽高，可以自定设定，默认不进行resize
# 注意：resize操作可能会使得图像处理的质量发生很大改变，若不想resize只需把new_width和new_height赋值为0

import cv2 as cv
import numpy as np

# 图像路径
image_path='原图路径'
groundtruth_path='groundTruth路径'

# resize scale
new_width = 0
new_height = 0


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

	# 形态学操作
	kr1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (40, 40))  # kernel，长轴和短轴均为40的椭圆，即半径20的圆
	img3 = cv.morphologyEx(img2, cv.MORPH_TOPHAT, kr1)  # 礼帽/顶帽运算
	kr2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (60, 60))
	img5 = cv.dilate(img3, kr2)  # 膨胀

	# 使用ground truth来保持病灶边缘的清晰
	gdt = cv.cvtColor(gdt, cv.COLOR_BGR2GRAY)  # 转换为灰度图
	t1, gdt = cv.threshold(gdt, 128, 255, cv.THRESH_BINARY_INV)  # 转换为反向2值图像
	img6 = cv.bitwise_and(img5, gdt)  # 进行逻辑与运算，得到hair mask
	return img6


def Method_1(origin_img, gdtruth):
	"""
	实现了论文方法一的函数，此函数将对原图像进行处理，并展示处理前后的对比图像
	:param gdtruth: 待处理图像对应的groundTruth图像
	:param origin_img: 待处理图像
	:return: none
	"""
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

	# display result
	# ------------------------------------------------------------------------------------------------------
	compare_img = np.concatenate((origin_img, result_img), 1)

	cv.namedWindow('Origin and Result', flags=0)
	cv.imshow('Origin and Result', compare_img)
	cv.waitKey(0)


if __name__=='__main__':
	# 读取图片
	img = cv.imread(image_path)  # 待处理的图片
	img_gdtruth = cv.imread(groundtruth_path)  # groundTruth图片
	# exception detecting
	if img.shape!=img_gdtruth.shape:
		raise Exception('groundtruth resolution is not match with the image resolution!')

	# resize图像大小，当图片过大时可使用resize来缩小图像，加快处理速度，resize会影响图像处理的效果，
	if new_height!=0 and new_width!=0:
		img=cv.resize(img, (new_width, new_height))
		img_gdtruth = cv.resize(img_gdtruth, (new_width, new_height))

	Method_1(img, img_gdtruth)
