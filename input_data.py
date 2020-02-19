import tensorflow as tf
import os 
import numpy as np

def get_files(file_dir):
    buildings = []
    label_buildings = []
    forest = []
    label_forest = []
    glacier = []
    label_glacier = []
    mountain = []
    label_mountain = []
    sea = []
    label_sea = []
    street = []
    label_street = []
    for files in os.listdir(file_dir):
        name = files.split(sep='.')
        if 'buildings' in name[0]:
            f = os.listdir(os.path.join(file_dir,files))
            site = os.path.join(file_dir,files)
            for file in f:
                buildings.append(os.path.join(site,file))
                label_buildings.append(0)
        if 'forest' in name[0]:
            f = os.listdir(os.path.join(file_dir,files))
            site = os.path.join(file_dir,files)
            for file in f:
                forest.append(os.path.join(site,file))
                label_forest.append(1)
        if 'glacier' in name[0]:
            f = os.listdir(os.path.join(file_dir,files))
            site = os.path.join(file_dir,files)
            for file in f:
                glacier.append(os.path.join(site,file))
                label_glacier.append(2)
        if 'mountain' in name[0]:
            f = os.listdir(os.path.join(file_dir,files))
            site = os.path.join(file_dir,files)
            for file in f:
                mountain.append(os.path.join(site,file))
                label_mountain.append(3)
        if 'sea' in name[0]:
            f = os.listdir(os.path.join(file_dir,files))
            site = os.path.join(file_dir,files)
            for file in f:
                sea.append(os.path.join(site,file))
                label_sea.append(4)
        if 'street' in name[0]:
            f = os.listdir(os.path.join(file_dir,files))
            site = os.path.join(file_dir,files)
            for file in f:
                street.append(os.path.join(site,file))
                label_street.append(5)
        image_list = np.hstack((buildings,forest,glacier,mountain,sea,street))
        label_list = np.hstack((label_buildings,label_forest,label_glacier,label_mountain,label_sea,label_street))
         
	# 把标签和图片都放倒一个 temp 中 然后打乱顺序，然后取出来
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
	# 打乱顺序
    np.random.shuffle(temp)

	# 取出第一个元素作为 image 第二个元素作为 label
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]  
    return image_list,label_list

# image_W ,image_H 指定图片大小，batch_size 每批读取的个数 ，capacity队列中 最多容纳元素的个数
def get_batch(image,label,image_W,image_H,batch_size,capacity):
	# 转换数据为 ts 能识别的格式
	image = tf.cast(image,tf.string)
	label = tf.cast(label, tf.int32)

	# 将image 和 label 放倒队列里 
	input_queue = tf.train.slice_input_producer([image,label])
	label = input_queue[1]
	# 读取图片的全部信息
	image_contents = tf.read_file(input_queue[0])
	# 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
	image = tf.image.decode_jpeg(image_contents,channels =3)
	# 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
	# 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
	image = tf.image.per_image_standardization(image)

	# 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
	image_batch, label_batch = tf.train.batch([image, label],batch_size = batch_size, num_threads = 64, capacity = capacity)
	
    # 重新定义下 label_batch 的形状
	label_batch = tf.reshape(label_batch , [batch_size])
	# 转化图片
	image_batch = tf.cast(image_batch,tf.float32)
	return  image_batch, label_batch