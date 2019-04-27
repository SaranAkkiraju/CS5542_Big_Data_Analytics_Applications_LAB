from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import glob
import os

from PIL import Image

cv_img = []

image_path = r"C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Lab3\show_and_tell\seg_images"
for img in glob.glob(r"C:\Users\praneeth\Documents\Studies\Big_Data_Analytics\Lab3\show_and_tell\images\*.jpg"):

    print(img)

    x =img[:img.find('.')]

    pic = plt.imread(img)/255  # dividing by 255 to bring the pixel values between 0 and 1

    print(pic.shape)

    plt.imshow(pic)

    pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])

    pic_n.shape

    kmeans = KMeans(n_clusters=8, random_state=0).fit(pic_n)

    pic2show = kmeans.cluster_centers_[kmeans.labels_]

    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])

    plt.imshow(cluster_pic)

    my_file = x.split('.')[0]+".jpg"
    
    try:
        plt.savefig(os.path.join(image_path, my_file)) 
    except:
        print("error")
    