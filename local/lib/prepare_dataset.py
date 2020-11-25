#import libraries

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
import tensorflow as tf

#read xml and get important items
def read_content(xml_file: str):    
    #open XML
    """
    Returns important information from a xml with a Bbox

    Args:
        xml_file (str): path of the xml you want to read.
    Returns:
        file_path (str): path of the image that corresponds to this xml.

        file_name (str): name of the image that corresponds to this xml.

        list_with_all_boxes (list): list whith 4 points of the Bbox. shape[N,4]

        list_with_all_names (list): list with labels of each Bbox. shape[N,1]
        
        list_with_image_dimentions: list with the shape of the image shape[1,1]
    """
    #open XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_all_names = []
    list_with_image_dimentions=[]

    #get inmportant information from XML
    for boxes in root.iter("object"):

        file_name = root.find("filename").text
        file_path = root.find("path").text
        name = boxes.find("name").text
        width = int(root.find("size").find("width").text)
        height= int(root.find("size").find("height").text)

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)


        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_all_names.append(name)
        list_with_image_dimentions.append([width,height])
        

    return file_path,file_name, list_with_all_boxes,list_with_all_names,list_with_image_dimentions


#create csv file with metadata or read csv with metadata 

def create_and_read_metadata(images_path="local\data\img", metadata_path="local\data"):
    
    if os.path.isfile(os.path.join(metadata_path,'metadata.csv'))==False:
    
#    get all paths in folder and create a list with path and name
    
        w=[]
        for file in os.listdir(images_path):
            if file.endswith(".xml"):
                w.append(read_content(os.path.join(images_path,file)))


        names,splits,labl,xmin,xmax,ymin,ymax=[],[],[],[],[],[],[]
        with open(os.path.join(metadata_path,'labels_map.json')) as f:
            dic = json.loads(f.read())


        for i in w:

            #split images in train and test
            split = np.random.rand(1)[0]
            if split<0.7:
                spl='train'
            else:
                spl='test'

            #create metadata columns
            splits.append(spl)
            labl.append(i[3][0])
            ymin.append(i[2][0][1]/i[4][0][1])
            ymax.append(i[2][0][3]/i[4][0][1])
            xmin.append(i[2][0][0]/i[4][0][0])
            xmax.append(i[2][0][2]/i[4][0][0])
            names.append(i[1])

        #generate dictionary to build csv
        df={}
        df["names"]=names
        df["xmin"]=xmin
        df["xmax"]=xmax
        df["ymin"]=ymin
        df["ymax"]=ymax
        df['split']=splits
        df['label']=labl
        df = pd.DataFrame(df)
        df=df.replace(dic)

        #create csv with metadata information
        df.to_csv(os.path.join(metadata_path,'metadata.csv'), index=False)
        metadata=df
    else:
        metadata=pd.read_csv(os.path.join(metadata_path,'metadata.csv'))
    return metadata

#function to separate train and test images
def build_sources_from_metadata(metadata, images_path="local\data\img", mode='train', exclude_labels=None): 

    """
    Returns a list with images paths and points of Bbox

    Args:
        metadata (DataFrame): Datafreame with metadata. shape[N,2].

        images_path (str): path with location of images.

        mode (str): "train" if you want to get the train dataset.
                    "test" if you want to get the test dataset.
        
        exclude_labels (list): list with labels you want to exclude 

    Returns:

        sources (list): list with path and Bbox points [["xmin"],["xmax"],df["ymin"],["ymax"]]. shape[N,2]
    """
    
    if exclude_labels is None:
        exclude_labels = set()
    if isinstance(exclude_labels, (list, tuple)):
        exclude_labels = set(exclude_labels)

    df = metadata.copy()
    df = df[df['split'] == mode]
    df['filepath'] = df['names'].apply(lambda x: os.path.join(images_path, x))
    include_mask = df['label'].apply(lambda x: x not in exclude_labels)
    df = df[include_mask]

    sources = list(zip(df['filepath'], zip(df["xmin"],df["xmax"],df["ymin"],df["ymax"])))
    return sources

#function to show a batch of three images

def imshow_batch_of_three(batch, show_box=True):
    
    """
    Returns plot of 3 images with bbox.

    Args:
        batch (iter): iter with loaded images.

        show_box (bol)= True if you want to print the Bbox
                        False if you dont want to print the Bbox

    Returns:

        plot 3 images of the dataset
    """

    boxes_batch = batch[1].numpy()
    image_batch = batch[0].numpy()
    fig, axarr = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    _,len_y,len_x,_=image_batch.shape

    for i in range(3):
        img = image_batch[i, ...]
        axarr[i].imshow(img)
        if show_box:
#             axarr[i].set(xlabel='cordenates = {}'.format(boxes_batch[i]))
            axarr[i].plot((int(boxes_batch[i][0]*len_x),int(boxes_batch[i][1]*len_x)),(int(boxes_batch[i][2]*len_y),int(boxes_batch[i][2]*len_y)),"r")
            axarr[i].plot((int(boxes_batch[i][0]*len_x),int(boxes_batch[i][1]*len_x)),(int(boxes_batch[i][3]*len_y),int(boxes_batch[i][3]*len_y)),"r")
            axarr[i].plot((int(boxes_batch[i][0]*len_x),int(boxes_batch[i][0]*len_x)),(int(boxes_batch[i][2]*len_y),int(boxes_batch[i][3]*len_y)),"r")
            axarr[i].plot((int(boxes_batch[i][1]*len_x),int(boxes_batch[i][1]*len_x)),(int(boxes_batch[i][2]*len_y),int(boxes_batch[i][3]*len_y)),"r")


#preprocessing images 

#reshape images
def preprocess_image(image,new_size=(227,227)):

    """
    Returns the image whith a different size.

    Args:
        batch (iter): iter with loaded images.

        show_box (bol)= True if you want to print the Bbox
                        False if you dont want to print the Bbox

    Returns:

        plot 3 images of the dataset
    """

    image = tf.image.resize(image, size=new_size)
    image = image / 255.0
    return image

def augment_image(image):

    return image



def make_tf_dataset(sources, training=False, batch_size=1,
    num_epochs=1, num_parallel_calls=1, shuffle_buffer_size=None):
    """
    Returns an operation to iterate over the dataset specified in sources

    Args:
        sources (list): A list of (filepath, label_id) pairs.
        training (bool): whether to apply certain processing steps
            defined only in training mode (e.g. shuffle).
        batch_size (int): number of elements the resulting tensor
            should have.
        num_epochs (int): Number of epochs to repeat the dataset.
        num_parallel_calls (int): Number of parallel calls to use in
            map operations.
        shuffle_buffer_size (int): Number of elements from this dataset
            from which the new dataset will sample.

    Returns:
        A tf.data.Dataset object. It will return a tuple images of shape
        [N, H, W, CH] and labels shape [N, 1].
    """
    def load(row):
        filepath = row['image']
        
        #read file
        img = tf.io.read_file(filepath)
        
        #tell TF this is an image jpg
        img = tf.io.decode_jpeg(img)
        
        return img, row['bbox']

    if shuffle_buffer_size is None:
        shuffle_buffer_size = batch_size*4

    images, boxes = zip(*sources)
    
    #create a TF dataset
    ds = tf.data.Dataset.from_tensor_slices({
        'image': list(images), 'bbox': list(boxes)}) 
    #line from the link. As with most code, if you remove an arbitrary line, expectin

    #shuffle dataset
    if training:
        ds = ds.shuffle(shuffle_buffer_size)
    #load images 
    ds = ds.map(load, num_parallel_calls=num_parallel_calls)
    
    #preprocces images
    ds = ds.map(lambda x,y: (preprocess_image(x), y), num_parallel_calls=num_parallel_calls)
    
    #data aumentation
    if training:
        ds = ds.map(lambda x,y: (augment_image(x), y))
        
    #repeat this order num_epochs times        
    ds = ds.repeat(count=num_epochs)
    #set size of the batch to return
    ds = ds.batch(batch_size=batch_size)
    #pre load x times of baches
    ds = ds.prefetch(1)

    return ds