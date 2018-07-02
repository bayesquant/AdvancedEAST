
# coding: utf-8

# In[1]:


import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
advEastPath='/data1/devuser/gpu/py/AdvancedEAST/'
sys.path.append(advEastPath)


# In[2]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image,ImageOps,ImageEnhance,ImageDraw

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import cfg
from label import point_inside_of_quad
from network import East
from preprocess import resize_image
from nms import nms


# In[3]:


east = East()
east_detect = east.east_network()
east_detect.load_weights('/data1/devuser/wty/src/AdvancedEAST/saved_model/east_model_weights_pre.h5')
# east_detect.load_weights(advEastPath+'model/weights_2T736.003-0.409.h5')
#east_detect.load_weights('/data1/devuser/wty/src/AdvancedEAST/saved_model/east_model_2T736.h5')


# In[ ]:


# ImageOps.autocontrast(image, cutoff=0, ignore=None)


# In[4]:


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))

def predict(inputImg):    
    pixel_threshold=0.7 #cfg.pixel_threshold
#     img = image.load_img(img_path)
    d_wight, d_height = resize_image(inputImg, cfg.max_predict_img_size)
    img = inputImg.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    if True:
        im=inputImg.copy()
        pixel_size=4
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * pixel_size
            py = (i + 0.5) * pixel_size
            line_width, line_color = 1, 'red'
            if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                if y[i, j, 2] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * pixel_size, py - 0.5 * pixel_size),
                       (px + 0.5 * pixel_size, py - 0.5 * pixel_size),
                       (px + 0.5 * pixel_size, py + 0.5 * pixel_size),
                       (px - 0.5 * pixel_size, py + 0.5 * pixel_size),
                       (px - 0.5 * pixel_size, py - 0.5 * pixel_size)],
                      width=line_width, fill=line_color)        
        quad_draw = ImageDraw.Draw(quad_im)
        quads = []
        for score, geo, s in zip(quad_scores, quad_after_nms,range(len(quad_scores))):
            if np.amin(score) > 0:
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=3, fill='blue')                
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()                
                quads.append(rescaled_geo_list)
        return im,quad_im,quads,y



# In[ ]:


# a,b,c,y=predict(image.load_img('/data1/devuser/wty/tmp/ori/14733375869835/000027.png'))
# b
#tt=np.transpose(np.array(c[0]).reshape(-1, 2))
# np.min(tt[0]),np.max(tt[0]),np.min(tt[1]),np.max(tt[1])
# int(max(tt[0]))


# In[5]:


import tesserocr
chi=tesserocr.PyTessBaseAPI()
chi.Init(lang='chi_sim')
chi.SetPageSegMode(tesserocr.PSM.RAW_LINE)
kor=tesserocr.PyTessBaseAPI()
kor.Init(lang='kor')
kor.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)
jpn=tesserocr.PyTessBaseAPI()
jpn.Init(lang='jpn')
jpn.SetPageSegMode(tesserocr.PSM.SINGLE_LINE)
# def rec(aa,threshold):
#     eng=[chi,kor,jpn]
#     def run_ocr(api):
#         api.SetImage(aa)
#         txt=api.GetUTF8Text()
#         conf=api.MeanTextConf()
#         return [txt,conf,api.GetInitLanguagesAsString()]
#     return list(filter(lambda x:x[1]>=threshold , map(run_ocr,eng)))
def rec1(aa,threshold):
    engList=[chi,kor,jpn]
    def run_ocr(img,api):
        api.SetImage(img)
        txt=api.GetUTF8Text()
        conf=api.MeanTextConf()
#         print(conf,api.GetInitLanguagesAsString())
        return [txt.replace(' ',' '),conf,api.GetInitLanguagesAsString(),img]
    images=[ImageOps.invert(aa),aa]
    return list(filter(lambda x:x[1]>=threshold , map(lambda api: max(map(lambda img:run_ocr(img,api),images),key=lambda l:l[1]) ,engList)))



# In[6]:


fname='/data1/devuser/wty/tmp/ori/14881852685932/000150.png'
fname='/data1/devuser/wty/tmp/ori/15143582091025/000210.png'
fname='/data1/devuser/wty/tmp/ori/15159978321567/000010.png'
fname='/data1/devuser/wty/tmp/ori/15159978321567/000110.png'
pic = cv2.imread(fname)
a,b,c,y=predict(Image.fromarray(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)))
# rst=detect(image)
# show(draw_illu(image.copy(), rst))
vv=[]
for t in c:
    tt=np.transpose(np.array(t).reshape(-1, 2)).astype(int)
    xx=tt[0]
    yy=tt[1]
    p1=pic[min(yy):max(yy),min(xx):max(xx)]
    a0=(Image.fromarray(cv2.cvtColor(p1,cv2.COLOR_BGR2RGB)))
    aa=ImageOps.autocontrast(a0, cutoff=10)
#     i1=Image.fromarray(p1)
#    aa=ImageEnhance.Contrast(i1).enhance(1.5)
#     vv.append(aa)
    print(str(min(yy))+":"+str(max(yy))+','+str(min(xx))+':'+str(max(xx)))
    rst=rec1(aa,60)
    if len(rst)>0:
        item=max(rst,key=lambda l:l[1])
        vv.append(item[3])
        print(item[:3] )
b


import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
def show(img):    
    plt.figure(figsize = (20,20))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:


_, axarr = plt.subplots(3, 3,figsize = (20,20))
for i in range(y.shape[2]):
    axarr[int(i/3),i%3].imshow(y[:,:,i])
    axarr[int(i/3),i%3].set_title('channel '+str(i))

