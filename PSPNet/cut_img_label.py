import os
import cv2
from PIL import Image

with open('train/test.txt','w') as f:
    for i in range(80):
        f.write(str(i)+'\n')
save = r'train/label'
images = os.listdir("E:\\AerialImageDataset\\train\\gt")
print(images)
pig = 0
for i, imag in enumerate(images):
    # print(type(i))
    # print('======', img, '======')
    img_name = imag
    img_orgin = cv2.imread("E:\\AerialImageDataset\\train\\gt" + "\\" + imag,0)
    img = cv2.resize(img_orgin, (512,512))
    h,w= img.shape
    stride = 256
    new_h,new_w = (h/stride+1)*stride,(w/stride+1)*stride
    print(new_h,new_w)
    img = cv2.resize(img,(int(new_w),int(new_h)))

    for i in range(h // stride):
        for j in range(w // stride):
            crop = img[i * stride:i * stride + 256, j * stride:j * stride + 256]
            crop[crop>120] = 255
            crop[crop<=120] = 0
            print(crop[crop>0])
            save_path = os.path.join(save,str(pig)+'.png')
            cv2.imwrite(save_path,crop)
            pig += 1