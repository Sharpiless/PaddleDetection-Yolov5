# PaddleDetection yolov5

[https://github.com/Sharpiless/PaddleDetection-Yolov5](https://github.com/Sharpiless/PaddleDetection-Yolov5)

# 简介

PaddleDetection飞桨目标检测开发套件，旨在帮助开发者更快更好地完成检测模型的组建、训练、优化及部署等全开发流程。

PaddleDetection模块化地实现了多种主流目标检测算法，提供了丰富的数据增强策略、网络模块组件（如骨干网络）、损失函数等，并集成了模型压缩和跨平台高性能部署能力。

经过长时间产业实践打磨，PaddleDetection已拥有顺畅、卓越的使用体验，被工业质检、遥感图像检测、无人巡检、新零售、互联网、科研等十多个行业的开发者广泛应用。

# Yolov5：

YOLOV4出现之后不久，YOLOv5横空出世。YOLOv5在YOLOv4算法的基础上做了进一步的改进，检测性能得到进一步的提升。虽然YOLOv5算法并没有与YOLOv4算法进行性能比较与分析，但是YOLOv5在COCO数据集上面的测试效果还是挺不错的。大家对YOLOv5算法的创新性半信半疑，有的人对其持肯定态度，有的人对其持否定态度。在我看来，YOLOv5检测算法中还是存在很多可以学习的地方，虽然这些改进思路看来比较简单或者创新点不足，但是它们确定可以提升检测算法的性能。其实工业界往往更喜欢使用这些方法，而不是利用一个超级复杂的算法来获得较高的检测精度。本文将对YOLOv5检测算法进行复现。

# 训练Yolov5：

```bash
python tools/train.py -c configs/yolov5/yolov5s_CSPdarknet_roadsign.yml
```

# 实验结果：

0.9087 mAP on roadsign dataset.

# 关注我的公众号：

感兴趣的同学关注我的公众号——可达鸭的深度学习教程：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210127153004430.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)


# 联系作者：

> B站：[https://space.bilibili.com/470550823](https://space.bilibili.com/470550823)

> CSDN：[https://blog.csdn.net/weixin_44936889](https://blog.csdn.net/weixin_44936889)

> AI Studio：[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156)

> Github：[https://github.com/Sharpiless](https://github.com/Sharpiless)


```python
%cd work/
```

    /home/aistudio/work



```python
!unzip PPDet-yolov5v2.zip -d ./
```


```python
!python tools/train.py -c configs/yolov5/yolov5s_CSPdarknet_roadsign.yml --eval
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/tensor/creation.py:125: DeprecationWarning: `np.object` is a deprecated alias for the builtin `object`. To silence this warning, use `object` by itself. Doing this will not modify any behavior and is safe. 
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      if data.dtype == np.object:
    [07/15 10:17:41] ppdet.utils.download WARNING: Config annotation dataset/roadsign_voc/train.txt is not a file, dataset config is not valid
    [07/15 10:17:41] ppdet.utils.download INFO: Dataset /home/aistudio/work/dataset/roadsign_voc is not valid for reason above, try searching /home/aistudio/.cache/paddle/dataset or downloading dataset...
    [07/15 10:17:41] ppdet.utils.download INFO: Found /home/aistudio/.cache/paddle/dataset/roadsign_voc/annotations
    [07/15 10:17:41] ppdet.utils.download INFO: Found /home/aistudio/.cache/paddle/dataset/roadsign_voc/images
    [07/15 10:17:41] reader WARNING: Shared memory size is less than 1G, disable shared_memory in DataLoader
    [07/15 10:17:42] ppdet.utils.checkpoint INFO: Finish loading model weights: output.pdparams
    [07/15 10:17:51] ppdet.engine INFO: Epoch: [0] [ 0/87] learning_rate: 0.000033 loss_xy: 0.752040 loss_wh: 0.698217 loss_iou: 2.634957 loss_obj: 11.301561 loss_cls: 1.041652 loss: 16.428429 eta: 8:28:32 batch_cost: 8.7679 data_cost: 0.9061 ips: 0.9124 images/s
    [07/15 10:19:42] ppdet.engine INFO: Epoch: [0] [20/87] learning_rate: 0.000047 loss_xy: 0.529626 loss_wh: 0.569290 loss_iou: 2.436198 loss_obj: 8.576855 loss_cls: 1.023474 loss: 13.317031 eta: 5:29:28 batch_cost: 5.5608 data_cost: 0.0002 ips: 1.4386 images/s
    [07/15 10:21:42] ppdet.engine INFO: Epoch: [0] [40/87] learning_rate: 0.000060 loss_xy: 0.500230 loss_wh: 0.502719 loss_iou: 2.226187 loss_obj: 4.208471 loss_cls: 0.890207 loss: 8.235611 eta: 5:35:40 batch_cost: 6.0032 data_cost: 0.0003 ips: 1.3326 images/s
    [07/15 10:23:23] ppdet.engine INFO: Epoch: [0] [60/87] learning_rate: 0.000073 loss_xy: 0.519860 loss_wh: 0.599364 loss_iou: 2.455585 loss_obj: 3.626266 loss_cls: 1.031202 loss: 8.345335 eta: 5:18:38 batch_cost: 5.0474 data_cost: 0.0003 ips: 1.5850 images/s
    [07/15 10:25:13] ppdet.engine INFO: Epoch: [0] [80/87] learning_rate: 0.000087 loss_xy: 0.568008 loss_wh: 0.618775 loss_iou: 2.583227 loss_obj: 3.632595 loss_cls: 0.863238 loss: 7.575019 eta: 5:15:29 batch_cost: 5.4984 data_cost: 0.0002 ips: 1.4550 images/s
    [07/15 10:25:47] ppdet.utils.checkpoint INFO: Save checkpoint: output/yolov5s_CSPdarknet_roadsign
    [07/15 10:25:47] ppdet.utils.download WARNING: Config annotation dataset/roadsign_voc/valid.txt is not a file, dataset config is not valid
    [07/15 10:25:47] ppdet.utils.download INFO: Dataset /home/aistudio/work/dataset/roadsign_voc is not valid for reason above, try searching /home/aistudio/.cache/paddle/dataset or downloading dataset...
    [07/15 10:25:47] ppdet.utils.download INFO: Found /home/aistudio/.cache/paddle/dataset/roadsign_voc/annotations
    [07/15 10:25:47] ppdet.utils.download INFO: Found /home/aistudio/.cache/paddle/dataset/roadsign_voc/images
    [07/15 10:25:48] ppdet.engine INFO: Eval iter: 0
    [07/15 10:26:09] ppdet.engine INFO: Eval iter: 100
    [07/15 10:26:25] ppdet.metrics.metrics INFO: Accumulating evaluatation results...
    [07/15 10:26:25] ppdet.metrics.metrics INFO: mAP(0.50, integral) = 85.84%
    [07/15 10:26:25] ppdet.engine INFO: Total sample number: 176, averge FPS: 4.751870228058035
    [07/15 10:26:25] ppdet.engine INFO: Best test bbox ap is 0.858.
    [07/15 10:26:25] ppdet.utils.checkpoint INFO: Save checkpoint: output/yolov5s_CSPdarknet_roadsign
    [07/15 10:26:35] ppdet.engine INFO: Epoch: [1] [ 0/87] learning_rate: 0.000091 loss_xy: 0.567437 loss_wh: 0.623783 loss_iou: 2.511684 loss_obj: 3.314124 loss_cls: 0.949793 loss: 7.338743 eta: 5:16:15 batch_cost: 6.2481 data_cost: 0.0003 ips: 1.2804 images/s
    [07/15 10:28:39] ppdet.engine INFO: Epoch: [1] [20/87] learning_rate: 0.000100 loss_xy: 0.583728 loss_wh: 0.708465 loss_iou: 2.704193 loss_obj: 3.461134 loss_cls: 1.127932 loss: 9.057523 eta: 5:20:59 batch_cost: 6.2270 data_cost: 0.0003 ips: 1.2847 images/s
    [07/15 10:30:28] ppdet.engine INFO: Epoch: [1] [40/87] learning_rate: 0.000100 loss_xy: 0.576615 loss_wh: 0.655194 loss_iou: 2.566234 loss_obj: 2.921384 loss_cls: 1.010778 loss: 7.844104 eta: 5:16:43 batch_cost: 5.4392 data_cost: 0.0003 ips: 1.4708 images/s
    [07/15 10:32:34] ppdet.engine INFO: Epoch: [1] [60/87] learning_rate: 0.000100 loss_xy: 0.583071 loss_wh: 0.726098 loss_iou: 2.730413 loss_obj: 3.053501 loss_cls: 0.991524 loss: 8.496977 eta: 5:19:40 batch_cost: 6.3128 data_cost: 0.0003 ips: 1.2673 images/s
    [07/15 10:34:31] ppdet.engine INFO: Epoch: [1] [80/87] learning_rate: 0.000100 loss_xy: 0.606061 loss_wh: 0.652358 loss_iou: 2.841094 loss_obj: 3.237591 loss_cls: 1.084277 loss: 8.605825 eta: 5:18:16 batch_cost: 5.8318 data_cost: 0.0003 ips: 1.3718 images/s
    [07/15 10:34:59] ppdet.utils.checkpoint INFO: Save checkpoint: output/yolov5s_CSPdarknet_roadsign
    [07/15 10:35:00] ppdet.engine INFO: Eval iter: 0
    [07/15 10:35:19] ppdet.engine INFO: Eval iter: 100
    [07/15 10:35:33] ppdet.metrics.metrics INFO: Accumulating evaluatation results...
    [07/15 10:35:33] ppdet.metrics.metrics INFO: mAP(0.50, integral) = 85.30%
    [07/15 10:35:33] ppdet.engine INFO: Total sample number: 176, averge FPS: 5.151774310709877
    [07/15 10:35:33] ppdet.engine INFO: Best test bbox ap is 0.858.
    [07/15 10:35:46] ppdet.engine INFO: Epoch: [2] [ 0/87] learning_rate: 0.000100 loss_xy: 0.537015 loss_wh: 0.587401 loss_iou: 2.352699 loss_obj: 3.121367 loss_cls: 1.012583 loss: 7.857001 eta: 5:17:11 batch_cost: 5.8271 data_cost: 0.0003 ips: 1.3729 images/s
    ^C



```python
!rm -rf output/
```


```python
!zip -r code.zip ./*
```

