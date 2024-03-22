import csv  
import os

# train images folder
train_HR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/div2k/DIV2K_train_HR'
train_LR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/div2k/DIV2K_train_LR_bicubic/X4'
# valid images folder
valid_HR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/div2k/DIV2K_valid_HR'
valid_LR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/div2k/DIV2K_valid_LR_bicubic/X4'
# test images folder
test_Set5_HR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/benchmark/Set5/HR'
test_Set5_LR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/benchmark/Set5/LR_bicubic/X4'
test_Set14_HR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/benchmark/Set14/HR'
test_Set14_LR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/benchmark/Set14/LR_bicubic/X4'
test_B100_HR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/benchmark/B100/HR'
test_B100_LR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/benchmark/B100/LR_bicubic/X4'
test_Urban100_HR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/benchmark/Urban100/HR'
test_Urban100_LR_folder = '/home/shaw/work/DL/hw7/code/SRDataset/benchmark/Urban100/LR_bicubic/X4'

output_folder = '/home/shaw/work/DL/hw7/code/SRDataset'

train_HR_images = []
train_LR_images = []
for i in range(800):
    image_num = '%04d' % (i+1)
    HR_image = image_num + '.png'
    LR_image = image_num + 'x4.png'
    train_HR_images.append(HR_image)
    train_LR_images.append(LR_image)
with open(os.path.join(output_folder, 'train_images.csv'), 'w') as j:
    writer = csv.writer(j)
    for i in range(800):
        writer.writerow([train_HR_images[i], train_LR_images[i]])

valid_HR_images = []
valid_LR_images = []
for i in range(100):
    image_num = '%04d' % (i+801)
    HR_image = image_num + '.png'
    LR_image = image_num + 'x4.png'
    valid_HR_images.append(HR_image)
    valid_LR_images.append(LR_image)
with open(os.path.join(output_folder, 'valid_images.csv'), 'w') as j:
    writer = csv.writer(j)
    for i in range(100):
        writer.writerow([valid_HR_images[i], valid_LR_images[i]])

Set5_HR_images = []
Set5_LR_images = []
images = os.listdir(test_Set5_HR_folder)
num = len(images)
for image in images:
    HR_image = image 
    LR_image = image.split('.')[0] + 'x4.png'
    Set5_HR_images.append(HR_image)
    Set5_LR_images.append(LR_image)
with open(os.path.join(output_folder, 'Set5_images.csv'), 'w') as j:
    writer = csv.writer(j)
    for i in range(num):
        writer.writerow([Set5_HR_images[i], Set5_LR_images[i]])

Set14_HR_images = []
Set14_LR_images = []
images = os.listdir(test_Set14_HR_folder)
num = len(images)
for image in images:
    HR_image = image 
    LR_image = image.split('.')[0] + 'x4.png'
    Set14_HR_images.append(HR_image)
    Set14_LR_images.append(LR_image)
with open(os.path.join(output_folder, 'Set14_images.csv'), 'w') as j:
    writer = csv.writer(j)
    for i in range(num):
        writer.writerow([Set14_HR_images[i], Set14_LR_images[i]])

B100_HR_images = []
B100_LR_images = []
images = os.listdir(test_B100_HR_folder)
num = len(images)
for image in images:
    HR_image = image 
    LR_image = image.split('.')[0] + 'x4.png'
    B100_HR_images.append(HR_image)
    B100_LR_images.append(LR_image)
with open(os.path.join(output_folder, 'B100_images.csv'), 'w') as j:
    writer = csv.writer(j)
    for i in range(num):
        writer.writerow([B100_HR_images[i], B100_LR_images[i]])

Urban100_HR_images = []
Urban100_LR_images = []
images = os.listdir(test_Urban100_HR_folder)
num = len(images)
for image in images:
    HR_image = image 
    LR_image = image.split('.')[0] + 'x4.png'
    Urban100_HR_images.append(HR_image)
    Urban100_LR_images.append(LR_image)
with open(os.path.join(output_folder, 'Urban100_images.csv'), 'w') as j:
    writer = csv.writer(j)
    for i in range(num):
        writer.writerow([Urban100_HR_images[i], Urban100_LR_images[i]])