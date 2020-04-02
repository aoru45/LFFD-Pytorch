'''
@Descripttion: This is Aoru Xue's demo,which is only for reference
@version: 
@Author: Aoru Xue
@Date: 2019-09-02 21:08:56
@LastEditors: Aoru Xue
@LastEditTime: 2019-10-03 00:22:04
'''
import torch
import torchvision
from model import BasketNet
from torchvision import transforms
#from transforms import *
from PIL import Image
import numpy as np
from viz import draw_bounding_boxes,draw_circles
from post_processer import PostProcessor
from priors import Priors

prior = Priors() 
center_form_priors = prior()

post_process = PostProcessor()
color_map= {
    1: (0,0,255),
    2: (0,255,255)
}
transform = transforms.Compose([
        transforms.Resize((640,640)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    )


def pic_test():
    img = Image.open("./test2.jpg").convert('RGB')
    image = np.array(img,dtype = np.float32)
    height, width, _ = image.shape
    img = transform(img)
    img = img.unsqueeze(0)
    #img = img.cuda()
    net = BasketNet()
    
    net.load_state_dict(torch.load("./ckpt/1748.pth",map_location="cpu"))
    #net.cuda()
    print("network load...")
    net.eval()
    with torch.no_grad():
        pred_confidence,boxes = net(img)
        
        
        output = post_process(pred_confidence,boxes, width=width, height=height)[0]
        #print(output)
        boxes, labels, scores = [o.to("cpu").numpy() for o in output]
        
        
        drawn_image = draw_bounding_boxes(image, boxes, labels, scores, ("__background__","basketball","volleyball")).astype(np.uint8)
       
        Image.fromarray(drawn_image).save("./a.jpg")

def cap_test():
    import cv2 as cv
    cap = cv.VideoCapture("../test.mp4")
    net = BasketNet()
    
    net.load_state_dict(torch.load("./ckpt/1748.pth",map_location="cpu"))
    #net.cuda()
    net.eval()
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        
        height,width,_ = frame.shape
        center_width = width//2
        frame = frame[:,center_width-height//2:center_width + height//2]
        height,width,_ = frame.shape
        cv_img = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        img = transform(img)
        img = img.unsqueeze(0)
        #img = img.cuda()

        with torch.no_grad():
            pred_confidence,boxes = net(img)
            
            output = post_process(pred_confidence,boxes, width=width, height=height)[0]
            boxes, labels, scores = [o.to("cpu").numpy() for o in output]
            drawn_image = draw_bounding_boxes(frame, boxes, labels, scores, ("__background__","basketball","volleyball")).astype(np.uint8)
            
            cv.imshow("img",drawn_image)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cv.destroyAllWindows()
    cap.release()
if __name__ == "__main__":
    pic_test()
