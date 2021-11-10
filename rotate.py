import os,cv2,numpy as np
import argparse
def file_name_listdir(file_dir,keyword=''):
    ret=[]
    for files in os.listdir(file_dir):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        if keyword=='':
            ret.append(os.path.join(file_dir,files))
        else:
            if files.find(keyword)!=-1:
                ret.append(os.path.join(file_dir,files))
    return ret

def rotate(image, angle):
    # rotate_bound_white_bg
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）

def increment_img(image,offset=1):
    image=np.clip(image.astype(int)+offset,0,255)
    return image.astype(np.uint8)
def identity(image,arg):
    return image

def process_folder(folder,func,arg=1,flip=False):
    img_list=file_name_listdir(folder)
    imagepath=img_list[0]
    print(img_list)
    if flip:
        img=cv2.flip(cv2.imread(img_list[0]),1)
    else:
        img=cv2.imread(img_list[0])
    for i in range(45):
        i1=func(img,arg)
        cv2.imwrite(os.path.join(folder,'{}.jpg').format(i*2+1),i1)
        cv2.imwrite(os.path.join(folder,'{}.jpg').format(i*2+2),img)
    cv2.imwrite(os.path.join(folder,'0.jpg'),img)



def clean_folder(folder):
    img_list=file_name_listdir(folder)
    img_list=img_list[1:]
    for img in img_list:
        os.remove(os.path.join(folder,img))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a series of images rotated from a source image.')
    parser.add_argument('-f','--folder',default='.',help='folder of target images')
    parser.add_argument('--clean',help='delete images of idx greater than 0',action="store_true")
    parser.add_argument('-f1','--func1',default='rotate',help='function 1')
    parser.add_argument('-a1','--arg1',default=0,type=int,help='arg 1')
    parser.add_argument('-f2','--func2',default='rotate',help='function 2')
    parser.add_argument('-a2','--arg2',default=0,type=int,help='arg 2')
    args=parser.parse_args()
    if not args.clean:
        process_folder(os.path.join(args.folder,'train'),eval(args.func1),args.arg1)
        process_folder(os.path.join(args.folder,'train_seg'),eval(args.func2),args.arg2)
    else:
        clean_folder(os.path.join(args.folder,'train'))
        clean_folder(os.path.join(args.folder,'train_seg'))
