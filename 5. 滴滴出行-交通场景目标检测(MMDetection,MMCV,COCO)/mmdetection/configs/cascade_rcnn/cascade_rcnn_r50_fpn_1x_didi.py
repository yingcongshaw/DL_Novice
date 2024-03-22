_base_ = [
    # 模型
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    # 数据集
    '../_base_/datasets/didi_detection.py',
    # 优化器
    '../_base_/schedules/schedule_1x_step3.py',
    #'../_base_/schedules/schedule_1x.py', 
    #'../_base_/schedules/schedule_1x_adam.py', 
    #'../_base_/schedules/schedule_1x_lr002.py', 
    #'../_base_/schedules/schedule_1x_20e.py', 
    # 训练方式 
    #'../_base_/default_runtime.py'
    '../_base_/runtime_batch_8.py'
]
# shaw