import os
import cv2
from PIL import Image

with open('train/test.txt','w') as f:
    for i in range(80):
        f.write(str(i)+'\n')
save = r'train/image'
images = os.listdir("E:\\AerialImageDataset\\train\\images")
print(images)
pig = 0
for i, imag in enumerate(images):
    # print(type(i))
    # print('======', img, '======')
    img_name = imag
    img_orgin = cv2.imread("E:\\AerialImageDataset\\train\\images" + "\\" + imag)
    img = cv2.resize(img_orgin, (1024,1024))
    h,w,_= img.shape
    stride = 512
    new_h,new_w = (h/stride+1)*stride,(w/stride+1)*stride
    print(new_h,new_w)
    img = cv2.resize(img,(int(new_w),int(new_h)))

    for i in range(h // stride):
        for j in range(w // stride):
            crop = img[i * stride:i * stride + 512, j * stride:j * stride + 512]
            save_path = os.path.join(save,str(pig)+'.png')
            cv2.imwrite(save_path,crop)
            pig += 1