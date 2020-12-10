# Skin-Hair-Removal-Based-On-Digital-Image-Processing
 一个毛发去除方法，以论文COMPARATIVE ANALYSIS OF AUTOMATIC SKIN LESION SEGMENTATION WITH TWO DIFFERENT IMPLEMENTATIONS作为参考来实现。

代码使用方法见代码文件中的注释

Hair_Removal.py为论文中Method 1预处理步骤中的毛发去除方法的复现

Hair_Removal_with_groundtruthimg.py是在Hair_Removal.py的基础上使用了groundTruth改进图像处理的质量

Hair_Removal_with_groundtruthimg_batch_processing.py是Hair_Removal_with_groundtruthimg.py的批量处理版本，可用于批量处理一个文件夹中的多个图像

Hair_Removal_with_groundtruthimg.py部分处理效果图如下所示：

![](https://i.loli.net/2020/12/10/5F1OPt7QElwvXKM.jpg)

![](https://i.loli.net/2020/12/10/RmJa5sdbLenMBFX.jpg)

![](https://i.loli.net/2020/12/10/j2O934xYUblsEpg.jpg)

Hair_Removal_with_groundtruthimg.py与Hair_Removal.py的结果对比，左侧图像为Hair_Removal.py的处理结果，右侧图像为Hair_Removal_with_groundtruthimg.py的处理结果

![](https://i.loli.net/2020/12/10/k6gHBP4GDlfCiS5.png)

![](https://i.loli.net/2020/12/10/FKICciTu83HN1EQ.png)

![](https://i.loli.net/2020/12/10/PxHQt7TbEFkZMon.png)