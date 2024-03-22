_base_ = [
    # 模型
    '../_base_/models/faster_rcnn_r50_fpn.py',
    # 数据集
    '../_base_/datasets/didi_detection.py',
    # 优化器
    '../_base_/schedules/schedule_1x.py',
    # 训练方式 
    '../_base_/default_runtime.py'
]