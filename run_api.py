import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/docs')
sys.path.append(os.getcwd()+'/docs/anti')
sys.path.append(os.getcwd()+'/docs/utils')
sys.path.insert(0,os.getcwd()+'/zypl_venv/')
sys.path.insert(0,os.getcwd()+'/zypl_venv/lib/python3.8/site-packages')
print(sys.path)
import random
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import numpy
import cv2
import Levenshtein
import pandas as pd
import re
import requests
import sys
import os
import easyocr
from ultralytics import YOLO
import shutil
import hashlib

from imutils.object_detection import non_max_suppression as non_max
import logging

import dlib
import time

import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from datetime import datetime

import warnings
from docs.parseq.main import read_parseq
from docs.anti.src.anti_spoof_predict import AntiSpoofPredict
from docs.anti.src.generate_patches import CropImage
from docs.anti.src.utility import parse_model_name

from models.common import DetectMultiBackend
from docs.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from docs.utils.general import (check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_segments, strip_optimizer, xyxy2xywh)
from docs.utils.plots import Annotator, colors, save_one_box
from docs.utils.torch_utils import select_device, time_sync
from deepface import DeepFace
from PIL.ExifTags import TAGS
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

warnings.filterwarnings('ignore')


ROOT = os.getcwd()

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

indices = ['А','Б','В','Г','Д','Е','Ё','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я','Ҷ','Қ','Ӣ','Ӯ','Ғ','Ҳ','а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я','ҷ','қ','ӣ','ӯ','ғ','ҳ','1','2','3','4','5','6','7','8','9','0','`',':','.',',','/','#','^','№','~','*',']','[','-','\\','|',chr(39),chr(34),'!','=','-',')','(',';','_']
reader_lang = easyocr.Reader(['ru','tjk'])
logging.getLogger('LOGGER').setLevel(logging.WARNING)


def check_exif(img):
    exifdata = img.getexif()
 
    if len(exifdata)>0:
        return True
    else:
        return False


# assert insightface.__version__ >= '0.3'

app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

def insight_model(img1, img2):
    img1 = cv2.imread(img1)
    faces1 = app.get(img1)

    feats1 = [face.normed_embedding for face in faces1]

    feats1 = np.array(feats1, dtype=np.float32)

    img2 = cv2.imread(img2)
    faces2 = app.get(img2)

    feats2 = [face.normed_embedding for face in faces2]

    feats2 = np.array(feats2, dtype=np.float32)

    sims = np.dot(feats1, feats2.T)

    try:
        similarity_percentage = round(sims[0][0] * 100, 2)
        return similarity_percentage
    except:
        return 0
    
SAMPLE_IMAGE_PATH = './docs'


def check_image(image):
    height, width, channel = image.shape

    return True


def is_fake(image_name, model_dir='./docs/anti/resources/anti_spoof_models', device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(image_name)
    result = check_image(image)
    if result is False:
        return
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            'org_img': image,
            'bbox': image_bbox,
            'scale': scale,
            'out_w': w_input,
            'out_h': h_input,
            'crop': True,
        }
        if scale is None:
            param['crop'] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time()-start

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2
    if label == 1:

        return ['Real', value]
    else:

        return ['Fake', value]
    
# from docs.parseq.main import read_parseq


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        read = []
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        all_read = []
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        #visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #save_path = str(save_dir / p.name)  # im.jpg
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_segments(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}{chr(115) * (n > 1)}, '  # add to string 
                # chr(115) = s

                # Write results
                read = []
                for *xyxy, conf, cls in reversed(det):
                    #if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #with open(f'{txt_path}.txt', 'a') as f:
                     #   f.write(('%g ' * len(line)).rstrip() % line + chr(10))
                    #print(('%g ' * len(line)).rstrip() % line)
                    read.append(('%g ' * len(line)).rstrip() % line)

                    #if save_img or save_crop or view_img:  # Add bbox to image
                     #   c = int(cls)  # integer class
                      #  label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                       # annotator.box_label(xyxy, label, color=colors(c, True))
                    #if save_crop:
                       # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    #if save_txt or save_img:
     #   s = f'{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}' if save_txt else ''
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    return read


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


#print('NEW version')
def detect(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))



path = 'docs'

model = YOLO(path + '/best_model1.pt')
# model1 = YOLO(path + '/best-y8m.pt')
model1 = YOLO(path + '/best_model5.pt')
model2 = YOLO(path + '/best_model2.pt')

find_phone_model = YOLO(path + '/best_model4.pt')

# def insight_model(img1, img2):
#     return 0.0
def convert_to_russian(text):
    new_text = ''
    text += ' '
    text = text.upper()
    to_read = False
    for i in range(len(text)):
        if not to_read:
            if text[i] == 'A':
                new_text += 'А'
                continue
            if text[i] == 'B':
                new_text += 'Б'
                continue
            if text[i] == 'C':
                try:
                    if text[i + 1] == 'H':

                        new_text += 'Ч'
                        if text[i + 2] == ' ':
                            return new_text
                        to_read = True
                        continue
                    if text[i + 1] == 'K':
                        new_text += 'К'
                        if text[i + 2] == ' ':
                            return new_text
                        to_read = True
                        continue
                except:
                    pass
                new_text += 'К'
                continue
            if text[i] == 'D':
                new_text += 'Д'
                continue
            if text[i] == 'E':
                if i == 0:
                    new_text += 'Э'
                    continue
                new_text += 'Е'
                continue
            if text[i] == 'F':
                new_text += 'Ф'
                continue
            if text[i] == 'G':
                try:
                    if text[i + 1] == 'H':

                        new_text += 'Ғ'
                        if text[i + 2] == ' ':
                            return new_text
                        to_read = True
                        continue
                except:
                    pass
                new_text += 'Г'
                continue
            if text[i] == 'H':
                new_text += 'Ҳ'
                continue
            if text[i] == 'I':
                new_text += 'И'
            if text[i] == 'J':
                new_text += 'Ҷ'
            if text[i] == 'K':
                try:
                    if text[i + 1] == 'H':

                        new_text += 'Х'
                        if text[i + 2] == ' ':
                            return new_text
                        to_read = True
                        continue
                except:
                    pass
                new_text += 'К'
            if text[i] == 'L':
                new_text += 'Л'
                continue
            if text[i] == 'M':
                new_text += 'М'
                continue
            if text[i] == 'N':
                new_text += 'Н'
                continue
            if text[i] == 'O':
                new_text += 'О'
                continue
            if text[i] == 'P':
                new_text += 'П'
                continue
            if text[i] == 'Q':
                new_text += 'Қ'
                continue
            if text[i] == 'R':
                new_text += 'Р'
                continue
            if text[i] == 'S':
                try:
                    if text[i + 1] == 'H':
                        new_text += 'Ш'
                        if text[i + 2] == ' ':
                            return new_text
                        to_read = True
                        continue
                except:
                    pass
                new_text += 'С'
                continue
            if text[i] == 'T':
                new_text += 'Т'
                continue
            if text[i] == 'U':
                new_text += 'У'
                continue
            if text[i] == 'V':
                new_text += 'В'
                continue
            if text[i] == 'W':
                new_text += 'В'
                continue
            if text[i] == 'X':
                new_text += 'КС'
                continue
            if text[i] == 'Y':
                try:
                    if text[i + 1] == 'A':
                        new_text += 'Я'
                        if text[i + 2] == ' ':
                            return new_text
                        to_read = True
                        continue
                    if text[i + 1] == 'U':
                        new_text += 'Ю'
                        to_read = True
                        continue
                    if text[i + 1] == 'O':
                        new_text += 'Ё'
                        to_read = True
                        continue
                except:
                    pass
                new_text += 'Й'
                continue
            if text[i] == 'Z':
                try:
                    if text[i + 1] == 'H':

                        new_text += 'Ж'
                        if text[i + 2] == ' ':
                            return new_text

                        to_read = True
                        continue
                except:
                    pass
                new_text += 'З'
                continue
        else:
            to_read = False

    return new_text



def clear_text(text, excepts):
    d = ''
    for i in text:
        i = i.replace('1', '')
        i = i.replace('2', '')
        i = i.replace('3', '')
        i = i.replace('4', '')
        i = i.replace('5', '')
        i = i.replace('6', '')
        i = i.replace('7', '')
        i = i.replace('8', '')
        i = i.replace('9', '')
        i = i.replace('0', '')
        i = i.replace(chr(39), '')
        i = i.replace(';', '')
        i = i.replace('-', '')
        i = i.replace('!', '')
        i = i.replace('@', '')
        i = i.replace('#', '')
        i = i.replace('%', '')
        i = i.replace('^', '')
        i = i.replace('&', '')
        i = i.replace('(', '')
        i = i.replace(')', '')
        i = i.replace('*', '')
        i = i.replace('_', '')
        i = i.replace('}', '')
        i = i.replace('{', '')
        i = i.replace('[', '')
        i = i.replace(']', '')
        i = i.replace('/', '')
        i = i.replace('?', '')
        i = i.replace('©', '')
        if excepts!=' ':
            i = i.replace(' ', '')
        i = i.replace('|', '')
        i = i.replace('.', '')
        i = i.replace(',', '')
        i = i.replace('»', '')
        i = i.replace('«', '')
        i = i.replace(chr(10), '')
        i = i.replace('	', '')
        i = i.replace('~', '')
        i = i.replace('—', '')
        i = i.replace('”', '')
        i = i.replace('=', '')
        i = i.replace('<', '')
        i = i.replace('>', '')
        i = i.replace(chr(34), '')
        i = i.replace('+', '')
        i = i.replace(':', '')
        i = i.replace('\\', '')
        i = i.replace(chr(12), '')
        i = i.replace(chr(10), '')
        i = i.replace('`','')
        d += i
    return d


ocr = PaddleOCR(lang='en', use_gpu=False, use_angle_cls=True,show_log = False)


def crop_pass_front(img):
    imgQ = cv2.imread(path + '/front_train.jpg', 0)
    h, w = imgQ.shape
    per = 14
    features = 14000
    orb = cv2.ORB_create(features)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    im = Image.open(img).convert('RGB')
    open_cv_image = numpy.array(im)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(img, None)
    matches = bf.match(des2, des1)
    matches1 = list(matches)
    matches1.sort(key=lambda x: x.distance)
    good = matches1[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    srcPoints = numpy.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = numpy.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    return imgScan

def tochka(date):
    date = date.replace('.', '')
    if len(date) == 8:
        date = date[0] + date[1] + '.' + date[2] + date[3] + '.' + date[4] + date[5] + date[6] + date[7]
    return date


def crop_fields(img):
    Surname_en = img[145:186, 270:550]
    Name_en = img[216:250, 270:550]
    Fathers_Name_en = img[284:316, 270:550]
    Date_of_Birth = img[342:371, 440:570]
    Date_of_Issue = img[392:422, 270:400]
    Date_of_Expiry = img[392:422, 440:570]
    Pass_Number = img[453:550, 10:250]
    Gender = img[339:367, 281:338]
    Nation = img[338:368, 343:446]
    Place_of_birth = img[343:370, 620:728]
    Nat_id = img[395:424, 617:786]
    return [Surname_en, Name_en, Fathers_Name_en, Date_of_Birth, Date_of_Issue, Date_of_Expiry, Pass_Number, Gender, Nation, Place_of_birth, Nat_id]



def read_paddle(fields):
    texts = []
    c=1
    for img in fields:
        text = ocr.ocr(img)
        try:
            texts.append(text[0][0][1][0])
        except:
            texts.append(' ')
        c+=1
    return texts

def replace_nation(text):
    if 'TJK' in text:
        return 'ТҶК/TJK'
    elif 'UZB' in text:
        return 'УЗБ/UZB'
    elif 'RUS' in text:
        return 'РУС/RUS'
    else:
        return text

def return_dict(letters):
    all_values = {'Surname_rus': convert_to_russian(letters[0]), 
                    'Surname_eng': letters[0], 
                    'Name_rus': convert_to_russian(letters[1]), 
                    'Name_eng': letters[1], 
                    'Fathers_Name_rus': convert_to_russian(letters[2]), 
                    'Fathers_Name_eng': letters[2],
                    'Date_of_birth': letters[3],
                    'Date_of_Issue': tochka(letters[4]),
                    'Date_of_Expiry': letters[5],
                    'Pass_num': letters[6],
                    'Gender': letters[7],
                    'Nation': replace_nation(letters[8]),
                    'Place_of_birth': replace_nation(letters[9]),
                    'Nat_id': letters[10]}

    return all_values


def read_paddle_gender(img):
    text = ocr.ocr(img)
    try:
        return text[0][0][1][0]
    except:
        return ' '


def clear_parseq(text):
    d = ''
    for i in text:
        i = i.replace('/','')
        i = i.replace('//','')
        i = i.replace(chr(39), '')
        i = i.replace(';', '')
        i = i.replace('-', '')
        i = i.replace('!', '')
        i = i.replace('@', '')
        i = i.replace('#', '')
        i = i.replace('%', '')
        i = i.replace('^', '')
        i = i.replace('&', '')
        i = i.replace('(', '')
        i = i.replace(')', '')
        i = i.replace('*', '')
        i = i.replace('_', '')
        i = i.replace('}', '')
        i = i.replace('{', '')
        i = i.replace('[', '')
        i = i.replace(']', '')
        i = i.replace('/', '')
        i = i.replace('?', '')
        i = i.replace('©', '')

        d += i
    return d

def read_passport(img, file_format='jpg'):
	random_r = random.randint(1,10000000000000)
	img2 = cv2.imread(img)

	if img2.shape[1]>1600 or img2.shape[0]>1600:
		h = img2.shape[1]
		w = img2.shape[0]
		prop = h/w

		new_w = 1200
		new_h = int(new_w*prop)
		front = cv2.resize(img2, (new_h, new_w))
	else:
		front = img2
	cv2.imwrite(path+f'/images/crop-pass-selfi.{file_format}', front)

    
	cropped_img = crop_pass_front(path + f'/images/crop-pass-selfi.{file_format}')
	fields = crop_fields(cropped_img)
	general = read_paddle(fields)
	dic = return_dict(general)

	return dic



def mn32(num):
    return num + (32 - num % 32)


def find_coords(arr, coor=[0, 0]):
    bbarr = arr.copy()
    n = len(bbarr)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if bbarr[j][2][1] > bbarr[j + 1][2][1]:
                bbarr[j], bbarr[j + 1] = bbarr[j + 1], bbarr[j]

    a = bbarr.copy()

    res = []
    mas = []
    for i in range(len(a) - 1):
        if a[i][2][1] * 1.2 + 3 >= a[i + 1][2][1]:
            if not a[i] in mas:
                mas.append(a[i])
            if not a[i + 1] in mas:
                mas.append(a[i + 1])
        else:
            if not a[i] in mas:
                mas.append(a[i])
            if len(mas) > 0:
                res.append(mas)
            mas = []
    if len(mas) > 0:
        res.append(mas)
    find = False
    for i in range(len(res)):
        if a[-1] in res[i]:
            find = True
            break
    if not find:
        res.append([a[-1]])

    arr = res.copy()


    for i in range(len(arr)):
        maxtopY = [arr[i][j][0][1] for j in range(len(arr[i]))]
        maxtopY = min(maxtopY)
        maxbottomY = [arr[i][j][1][1] for j in range(len(arr[i]))]
        maxbottomY = max(maxbottomY)

        for j in range(len(arr[i])):
            arr[i][j][0][1] = maxtopY
            arr[i][j][1][1] = maxbottomY

    for i in range(len(arr) - 1):
        if arr[i][0][1][1] >= arr[i + 1][0][0][1]:
            diff = arr[i][0][1][1] - arr[i + 1][0][0][1]
            for w in range(len(arr[i])):
                arr[i][w][1][1] = arr[i][w][1][1] - diff // 2
            for w in range(len(arr[i + 1])):
                arr[i + 1][w][0][1] = arr[i + 1][w][0][1] + diff // 2

    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j][0][0] < coor[0] < arr[i][j][1][0]:
                if arr[i][j][0][1] < coor[1] < arr[i][j][1][1]:
                    return bbarr.index(arr[i][j]) + 1
    return -1


def finder(image):
    min_conf = 0.5
    coords = []
    width = mn32(image.shape[1])
    height = mn32(image.shape[0])
    (H, W) = image.shape[:2]
    (newW, newH) = (width, height)
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    layerNames = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
    net = cv2.dnn.readNet(path + '/best_model3.pb')
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(0, numCols):
            if scoresData[x] < min_conf:
                continue
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    boxes = non_max(np.array(rects), probs=confidences)
    for x, (startX, startY, endX, endY) in enumerate(boxes):
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        coor = [[startX, startY], [endX, endY], [(startX + endX) // 2, (startY + endY) // 2]]
        coords.append(coor)
    return coords


def say_box(image, coords):
    boxes = []
    
    Xsize = image.shape[1]
    Ysize = image.shape[0]
    coords = coords.split(chr(10))
    fcoords = finder(image)
    for i in coords:
        if len(i) == 0:
            continue
        this = i.split()
        startX = int(Xsize * float(this[1]))
        startY = int(Ysize * float(this[2]))
        f = find_coords(fcoords, [startX, startY])
        boxes.append(f)
    return boxes


def bubble_sort(our_list, list2 = [], list3 = [],list4 = [], list5 = [],list6 = []):
    for i in range(len(our_list)):
        for j in range(len(our_list) - 1):
            if our_list[j] > our_list[j + 1]:
                
                our_list[j], our_list[j + 1] = our_list[j + 1], our_list[j]
                
                try:
                  list2[j], list2[j + 1] = list2[j + 1], list2[j]
                except:
                  pass
                try:
                  list3[j], list3[j + 1] = list3[j + 1], list3[j]
                except:
                  pass
                try:
                  list4[j], list4[j + 1] = list4[j + 1], list4[j]
                except:
                  pass
                try:
                  list5[j], list5[j + 1] = list5[j + 1], list5[j]
                except:
                  pass

                try:
                  list6[j], list6[j + 1] = list6[j + 1], list6[j]
                except:
                  pass
    return our_list, list2, list3,list4,list5,list6



def split_lines(image,txt='', ilist = []):
  lines = []

  
  space_list = say_box(image, txt)
  txt = txt.split(chr(10))

  line = []
  
  labels = []
  xs = []
  ys = []
  x_wid = []
  y_wid = []
  for i in range(len(txt)):
    text  =  txt[i].split()
    labels.append(text[0])
    xs.append(float(text[1]))
    ys.append(float(text[2]))
    x_wid.append(float(text[3]))
    y_wid.append(float(text[4]))

  
  ys,labels,xs,x_wid,y_wid,space_list = bubble_sort(ys,labels,xs,x_wid,y_wid,space_list)
  word_space = []

  word = [space_list[0]]
  line.append([labels[0],xs[0]])
  for i in range(1,len(ys)):

    if ys[i]-ys[i-1]<0.015:
      line.append([labels[i],xs[i]])
      word.append(space_list[i])
    if ys[i]-ys[i-1]>=0.015 or i==len(ys)-1:
      lines.append(line)
      line = []
      word_space.append(word)
      word = []
      word.append(space_list[i])
      line.append([labels[i],xs[i]])
  
  ans = []
  all_x = []
  widths = []
  count = -1
  new_word_space = []
  for line in lines:
    count+=1
    labels = []
    xs = []
    for i in range(len(line)):
      labels.append(line[i][0])
      xs.append(line[i][1])

    for i in range(len(labels)):
      
      labels[i] = ilist[int(labels[i])]
    xs, labels,x_wid,word_space[count],a ,s= bubble_sort(xs,labels,x_wid,word_space[count])

    label = []
    X = []
    width = []
    words = []
    for i in range(len(labels)):
      
      try:

        if labels[i]=='3' and labels[i+1]=='З':
          labels.pop(i+1)
          xs.pop(i+1)
          word_space[count].pop(i+1)
          x_wid.pop(i+1)
        if labels[i]=='З' and labels[i+1]=='3':
          labels.pop(i)
          xs.pop(i)
          word_space[count].pop(i)
          x_wid.pop(i)
        if labels[i]==labels[i+1]:
          continue

        if labels[i]=='0' and labels[i+1].upper()=='O':


          word_space[count].pop(i)    
          labels.pop(i)
          xs.pop(i)
              
        if labels[i].upper()=='O' and labels[i+1]=='0':
          word_space[count].pop(i+1) 
          labels.pop(i+1)
          xs.pop(i+1)

          
        if labels[i]=='Ҳ' and labels[i+1]=='Х':
          word_space[count].pop(i)
          labels.pop(i)
          xs.pop(i)
          
        if labels[i]=='Х' and labels[i+1]=='Ҳ':
          word_space[count].pop(i+1) 
          labels.pop(i+1)
          xs.pop(i+1)

        if labels[i]=='И' and labels[i+1]=='Ӣ':
          word_space[count].pop(i)
          labels.pop(i)
          xs.pop(i)
          
        if labels[i]=='Ӣ' and labels[i+1]=='И':
          word_space[count].pop(i+1) 
          labels.pop(i+1)
          xs.pop(i+1)

        if labels[i]=='Ғ' and labels[i+1]=='Г':
          word_space[count].pop(i)
          labels.pop(i)
          xs.pop(i)
          
        if labels[i]=='Г' and labels[i+1]=='Ғ':
          word_space[count].pop(i+1) 
          labels.pop(i+1)
          xs.pop(i+1)

        if labels[i]=='Ч' and labels[i+1]=='Ҷ':
          word_space[count].pop(i)
          labels.pop(i)
          xs.pop(i)
          
        if labels[i]=='Ҷ' and labels[i+1]=='Ч':
          word_space[count].pop(i+1) 
          labels.pop(i+1)
          xs.pop(i+1)

        if labels[i]=='Қ' and labels[i+1].upper()=='К':
          word_space[count].pop(i)
          labels.pop(i)
          xs.pop(i)
          
        if labels[i].upper()=='К' and labels[i+1]=='Қ':
          word_space[count].pop(i+1) 
          labels.pop(i+1)
          xs.pop(i+1)

        if labels[i]=='Ё' and labels[i+1]=='Е':
          word_space[count].pop(i)
          labels.pop(i)
          xs.pop(i)
          
        if labels[i]=='Е' and labels[i+1]=='Ё':
          word_space[count].pop(i+1)
          labels.pop(i+1)
          xs.pop(i+1)  
        
        if labels[i]=='И' and labels[i+1].upper()=='Й':
          word_space[count].pop(i)
          labels.pop(i)
          xs.pop(i)
          
        if labels[i].upper()=='Й' and labels[i+1]=='И':
          word_space[count].pop(i+1) 
          labels.pop(i+1)
          xs.pop(i+1)
        
        if not labels[i].isdigit():
          if labels[i]==labels[i+1]:
            word_space[count].pop(i+1)     
      except:
        pass

      try:

        label.append(labels[i])
        X.append(xs[i])
        width.append(x_wid[i]/8)
        words.append(word_space[count][i])
      except:
        pass
      
    new_word_space.append(words)  
    all_x.append(X)
    ans.append(label)
    widths.append(width)
  
  for i in range(len(ans)):
    answer = ''
    for j in ans[i]:
      answer+=j
    
    ans[i]=answer
  new_mes = ''
  for i in range(len(ans)):
     new_mes+=ans[i][0]
     for j in range(1,len(ans[i])):
        if new_word_space[i][j]!=new_word_space[i][j-1]:
           new_mes+=' '
        new_mes+=ans[i][j]
     new_mes+=chr(10)

  tt = open(path + '/test.txt', 'w',encoding='utf-8')
  tt.write(new_mes)
  tt.close()

  f = open(path + '/test.txt', 'r',encoding='utf-8')
  txt = f.readlines()
  f.close()

  new_mes1 = ''
  for i in txt:
     if len(i)>2:
        new_mes1+=i
  
  return new_mes1


def read_yolo1(img):
  confi = .3
  results = model1.predict(img,conf = confi, save_txt=True)
  result = results[0]
  filename = img.split('/')[-1]

  file = filename.replace('jpg','txt')  

  dirname = result.save_dir + '/labels/' + file

  f = open(dirname, 'r', encoding='utf-8')
  txt = f.read()
  f.close()
  
  return txt


def read_yolo2(img):
  results = model2.predict(img, save_txt=True)
  result = results[0]
  filename = img.split('/')[-1]

  file = filename.replace('jpg','txt')

  dirname = result.save_dir + '/labels/' + file
  
  f = open(dirname, 'r', encoding='utf-8')
  txt = f.read()
  f.close()
  
  return txt


def reader_yolo1(img):

    r = read_yolo1(img)

    split = r.split(chr(10))
    del split[-1]
    update_txt = chr(10).join(split)

    read = split_lines(cv2.imread(img),update_txt, indices)

    r1=read.replace('. ','.')
    r2 = r1.replace(' .','.')
    r3 = r2.replace(' ,',',')

    
    return r3


def reader_yolo2(img):

    r = read_yolo2(img)

    split = r.split(chr(10))
    del split[-1]
    update_txt = chr(10).join(split)

    read = split_lines(cv2.imread(img),update_txt, indices)
    r1=read.replace('. ','.')
    r2 = r1.replace(' .','.')
    r3 = r2.replace(' ,',',')

    
    return r3


def split_n(text):
  split = text.split()

  remove_list = []
  for i in split:
   if len(i)<4:
      if 'Ҳ' in i or 'Ҷ' in i or 'Қ' in i or 'Ӣ' in i or 'Ғ' in i or 'Ӯ' in i:
         remove_list.append(i)
  
  result_list = [item for item in split if item not in remove_list]
  return result_list


def filt(text):
    file = open(path + '/new_add.txt','r', encoding='utf-8')
    txt = file.readlines()
    file.close()
    file_datas = []

    for h in txt:
        file_datas.append(h.replace(chr(10),''))
    word_list=file_datas
    result = []
    for word in text.split():
        min_distance = float('inf')
        closest_word = word
        for list_word in word_list:
            distance = Levenshtein.distance(word, list_word)
            if distance == 1:
                closest_word = list_word
        result.append(closest_word)

    last_ans = ' '.join(result)
    replacer1 = last_ans.replace(',,',',')
    replacer2 = replacer1.replace('..','.')

    return replacer2


def filt1(text):
  tesseract_text = text.split()
  word_to_remove = ['ВИЛОЯТИ', 'ШАҲРИ', 'ШАҲРУ']

  new_lis = []
  trigger = True
  for i in range(len(tesseract_text)):
    dist=20
    for j in range(len(word_to_remove)):

      dis = Levenshtein.distance(tesseract_text[i], word_to_remove[j])
      if dis<dist:
        dist=dis

    if dist<3:
      tex = tesseract_text[i:]
      trigger = False
      tesseract_text = ' '.join(tex)  


  b= ''.join(tesseract_text)
  if trigger:
    return text      
  return b


def labeling(img):

    imgname = img
    org = [run(weights=path + '/best_model1.pt', conf_thres=0.38, source=imgname)]
    #org1 = [detect.run(weights='bdm.pt', conf_thres=0.3 8, source=imgname)]

    mes=[]
    for i in org[0]:
        mes.append(i)

    inpu = ''
    for hh in mes:
        inpu+=hh[0:len(hh)-1]+chr(10)

    return inpu


def reader(img):

    r = labeling(img)

    split = r.split(chr(10))
    del split[-1]
    update_txt = chr(10).join(split)

    read = split_lines(cv2.imread(img),update_txt, indices)

    r1=read.replace('. ','.')
    r2 = r1.replace(' .','.')
    r3 = r2.replace(' ,',',')

      
    return r3

def reader_address(img):

    text = reader_yolo1(img)
    text3 = reader_yolo2(img)

    model1 = text
    new_m1=''
    for i in range(1,len(model1)-1):
        if model1[i].isupper():

            if model1[i].lower()==model1[i+1]:
                continue

            if model1[i].lower()==model1[i-1]:
                continue
        new_m1 += model1[i]

    new_m1 = filt(filt1(new_m1))
    model4 = filt(filt1(text3))
    
    shutil.rmtree('runs/')
    return [new_m1, model4]
    
    
def read_issue(img):
    text = reader(img)
    model1 = filt(text)
    return model1
    

def crop_pass_back(img):
    imgQ = cv2.imread(path + '/back_train.jpg', 0)
    h, w = imgQ.shape
    per = 25
    features = 14500
    orb = cv2.ORB_create(features)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    im = Image.open(img).convert('RGB')
    open_cv_image = numpy.array(im)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(img, None)
    matches = bf.match(des2, des1)
    matches1 = list(matches)
    matches1.sort(key=lambda x: x.distance)
    good = matches1[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    srcPoints = numpy.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = numpy.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    return imgScan


def showlbl(key,img):
    img=cv2.imread(img)
    rt={'marital': [[0.4277511961722488, 0.36882716049382713], [0.5684210526315789, 0.42592592592592593]],
        'issuing_authority': [[0.2851674641148325, 0.44907407407407407], [0.9674641148325359, 0.6033950617283951]],
        'tax_payer': [[0.7913875598086124, 0.3734567901234568], [0.9885167464114832, 0.4444444444444444]],
        'address': [[0.2851674641148325, 0.038580246913580245], [0.9741626794258373, 0.30246913580246915]]
    }
    r=rt[key]
    x1,x2=int(img.shape[1]*r[0][0]),int(img.shape[1]*r[1][0])
    y1,y2=int(img.shape[0]*r[0][1]),int(img.shape[0]*r[1][1])
    img=img[y1:y2,x1:x2]
    return img


def marital_status(string):
    ret_text = ''
    if 'MARRIED' in string:
        ret_text = 'ОИЛАДОР'
    if 'SINGLE' in string:
        ret_text = 'МУҶАРРАД'

    return ret_text



def filter_issue(text):
    s=text
    spl = s.split()
    for i in range(1,10):
        if str(i) in spl[-1]:
            spl.remove(spl[-1])

    re = ' '.join(spl)
    ans = re.replace('ШВК', 'ШВКД')
    return ans

def split_numbers(text):
  ld = False

  new_txt = ''

  for i in range(len(text)):
    
    if text[i].isdigit():
      if ld==False:
        new_txt+=' '
        ld = True
      
    else:
      if ld==True:
        new_txt+=' '
        ld = False
      
    new_txt+=text[i]
  return new_txt



def levenshtein(img):
    img_s = cv2.imread(img)
    

    if img_s.shape[0]>1800 or img_s.shape[1]>1800:
        h = img_s.shape[1]
        w = img_s.shape[0]
        prop = h/w

        new_w = 1200
        new_h = int(new_w*prop)
        
        resize = cv2.resize(img_s, (new_h, new_w))
        cv2.imwrite(path + '/images/image-resized.jpg', resize)
        cv2.imwrite(path + '/images/crop.jpg', crop_pass_back(path + '/images/image-resized.jpg'))
    else:
        cv2.imwrite(path + '/images/crop.jpg', crop_pass_back(img))

    cv2.imwrite(path + '/images/crop-marital.jpg', showlbl('marital', path + '/images/crop.jpg'))
    cv2.imwrite(path + '/images/crop-issuing_authority.jpg', showlbl('issuing_authority', path + '/images/crop.jpg'))
    cv2.imwrite(path + '/images/crop-tax_payer.jpg', showlbl('tax_payer', path + '/images/crop.jpg'))
    cv2.imwrite(path + '/images/crop-address.jpg', showlbl('address', path + '/images/crop.jpg'))

    labels = ['marital', 'tax_payer']

    lis = []
    for i in labels:
        result = ocr.ocr(path + f'/images/crop-{i}.jpg', det=False, cls=True)[0][0][0]
        lis.append(result)

    
    if len(lis[1])!=9:
        tax_read = read_parseq(path + '/images/crop-tax_payer.jpg')
    else:
        tax_read = lis[1]

        
    final_list=[marital_status(lis[0]), tax_read]

    try:
        iss_aut = read_issue(path + '/images/crop-issuing_authority.jpg').upper()
    except:
        iss_aut = 'None'


    
    address = reader_address(path + '/images/crop-address.jpg')

    address1 = address[0]
    # address2 = address[2]
    # address3 = address[2]
    address4 = address[1]

    issue = filter_issue(iss_aut).replace('ШВКДД', 'ШВКД')
    issue1 = issue.replace('НОҲИЯ', 'НОҲИЯИ')
    issue2 = issue1.replace('ШДУШАНБЕ', 'Ш.ДУШАНБЕ')
    issue3 = issue2.replace('НОҲИЯИИ', 'НОҲИЯИ')

    dic={
        'Address1': address1,
        'Address': address4,
        'Marital_status': final_list[0],
        'Tax_payer_ID_number': final_list[1],
        'Issuing_Authority': issue3.replace('ШАҲРУ', 'ШАҲРИ')
    }
    return dic


def recognition(front, back):
    front_result = read_passport(front)
    back_result = levenshtein(back)

    res = {
        'Front': front_result,
        'Back': back_result
    }



    return res



def find_phone(img):
    results = find_phone_model(img, save_txt=True)  # predict on an image

    filename = img.split('/')[-1]

    r1 =filename.replace('jpg', 'txt')
    r2 = r1.replace('png', 'txt')

    try:
        f = open(results[0].save_dir + '/labels/' + r2, 'r')
        txt = f.readlines()
        f.close()
        new = []
        for i in txt:
            new.append(int(i.split()[0]))

        dd = []
        for j in range(len(new)):
            d = results[0].names[new[j]]
            dd.append(d)


        if 'cell phone' in dd:
            return True
        else:
            return False
    except:
        return False





def big_face_area(img_path):
    try:
        areas = []
        faces = DeepFace.represent(img_path, detector_backend='retinaface', model_name='Facenet512')

        ok=[]
        for i in faces:
            i=i['facial_area']
            ok.append(i)
            areas.append(i['w']*i['h'])

        cod = faces[areas.index(min(areas))]['facial_area']

        area = [cod['x'], cod['y'], cod['w'], cod['h']]
        img = cv2.imread(img_path)

        image = cv2.rectangle(img, (area[0], area[1]), (area[0]+area[2], area[1]+area[3]), color = (255, 0, 0), thickness=-1)

        cv2.imwrite(path + '/images/crop-selfi-bigface.jpg', image)
        return image
    except:
        cv2.imwrite(path + '/images/crop-selfi-bigface.jpg', img_path)


def proverka_pass_num(number = ''):
    leng = len(number)

    if leng==9 and number[0]!='A':
        return 'A'+number[1:8]
    elif leng==8 and number[0]!='A':
        return 'A'+number
    elif leng==9 and number[0]=='A':
        return number
    else:
        return number

def count_faces(img_path):
    detector = dlib.get_frontal_face_detector()

    image = cv2.imread(img_path)
    res = cv2.resize(image, (image.shape[1]*2, image.shape[0]*2))

    faces1 = detector(image)
    num_faces1 = len(faces1)

    faces2 = detector(res)
    num_faces2 = len(faces2)
    return max(num_faces1, num_faces2)

    
def read_pn(img):
    img1 = img[453:550, 10:250]
    result = ocr.ocr(img1, det=False, cls=True)

    return result[0][0][0]
#1589)

def crop_img(img1 ,per=14, features=14000):
    imgQ = cv2.imread(path + '/front_train.jpg', 0)
    h, w = imgQ.shape
    per = per
    features = features
    orb = cv2.ORB_create(features)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    if type(img1)==type('-'):
        im = Image.open(img1).convert('RGB')
    else:
        im=Image.fromarray(img1).convert('RGB')
    open_cv_image = numpy.array(im)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(img, None)
    matches = bf.match(des2, des1)
    matches1 = list(matches)
    matches1.sort(key=lambda x: x.distance)
    good = matches1[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    srcPoints = numpy.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = numpy.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    return imgScan

def crop_img2(img1 ,per=1, features=14000):
    imgQ = cv2.imread(path + '/front_train2.jpg', 0)
    h, w = imgQ.shape
    per = per
    features = features
    orb = cv2.ORB_create(features)
    kp1, des1 = orb.detectAndCompute(imgQ, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    if type(img1)==type('-'):
        im = Image.open(img1).convert('RGB')
    else:
        im=Image.fromarray(img1).convert('RGB')
    open_cv_image = numpy.array(im)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(img, None)
    matches = bf.match(des2, des1)
    matches1 = list(matches)
    matches1.sort(key=lambda x: x.distance)
    good = matches1[:int(len(matches) * (per / 100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)
    srcPoints = numpy.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = numpy.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    return imgScan

def read_pn2(img):
    resize_n = 2
    img = img[453//resize_n:550//resize_n, 10//resize_n:250//resize_n]
    result = ocr.ocr(img, det=False, cls=True)
    return result[0][0][0]


def is_flip(img):
    now = time.time()
    try:
        crop_front = crop_img(img)
        crop_front_small = crop_img2(img)
    except:
        return {'Pass_match': 'None', 'Face_match': 'None', 'Error': 'Couldn`t crop passport on front'}
    cv2.imwrite(path + '/images/crop-img.png', crop_front)

    front_pass_num = read_pn(crop_front)
    front_pass_num_small = read_pn2(crop_front_small)


    front_img = cv2.imread(img)
    flip_img = cv2.flip(front_img, 1)
    cv2.imwrite(path + '/images/flip-image.png', flip_img)
    cv2.imwrite(path + '/images/flip-image.jpg', flip_img)

    try:
        crop_flip = crop_img(path + '/images/flip-image.png')
        crop_flip_small = crop_img2(path + '/images/flip-image.png')

    except:
        return {'Pass_match': 'None', 'Face_match': 'None', 'Error': 'Couldn`t flip image'}

    cv2.imwrite(path + '/images/crop-flip.jpg', crop_flip)
    cv2.imwrite(path + '/images/crop-flip-small.jpg', crop_flip_small)
    flip_pass_num = read_pn(crop_flip)
    flip_pass_num_small = read_pn2(crop_flip_small)
    if len(front_pass_num)>=6 or len(front_pass_num_small)>=6:
        return False
    elif len(flip_pass_num)>=6 or len(flip_pass_num_small)>=6:
        return True


def read_yolo(img):
  try:
    results = model.predict(img, save_txt=True)
    result = results[0]
    filename = img.split('/')[-1]
    file = filename.replace('png','txt')
    file1 = filename.replace('jpg','txt')
    dirname = result.save_dir + '/labels/' + file1
    f = open(dirname, 'r', encoding='utf-8')
    txt = f.readlines()
    f.close()
    if len(txt)>0:
        return True
    else:
        return False
  except:
       return False
  
def is_PassNum(text):
    if len(text)>=3:
        return True
    else:
        return False

def front_verification(img1_p = 'None'):
    result = {}

    try:
        front_img = cv2.imread(img1_p)
        if front_img.shape[1]>2500 and front_img.shape[0]>2500:
            front = cv2.resize(front_img, (front_img.shape[1]//2, front_img.shape[0]//2))
            cv2.imwrite(path + '/images/crop-pass.jpg', front)
        else:
            front = cv2.imread(img1_p)
            cv2.imwrite(path + '/images/crop-pass.jpg', front)


    except:
        return {'Pass_match': 'None', 'Face_match': 'None', 'Error': 'Couldn`t read front image'}
    
    result['Find_phone'] = find_phone(img1_p)
    try:
        size = ''
        crop_front = crop_img(path + '/images/crop-pass.jpg')
        front_pass_num = read_pn(crop_front)
        if len(front_pass_num)>=5:
            size='big'
        else:
            size='small'

        if size=='big':
            front_pass_num = front_pass_num
            check_passNum = is_PassNum(front_pass_num)
            cv2.imwrite(path + '/images/crop-front.jpg', crop_front)
            cv2.imwrite(path + '/images/crop-front.png', crop_front)

            if check_passNum == True:
                result['is_Pass_full'] = 'Done'
                result['find_passNum_on_front'] = 'Done'
            else:
                check_passNum = False
                check_passNum_small = False
                result['is_Pass_full'] = 'Pass isn`t full'
                result['find_passNum_on_front'] = 'Couldn`t find PassNum'

        else:
            crop_front_small = crop_img2('verification_img/crop-pass.jpg')
            front_pass_num_small = read_pn2(crop_front_small)
            front_pass_num = front_pass_num_small
            check_passNum_small = is_PassNum(front_pass_num_small)
            cv2.imwrite(path + '/images/crop-front-small.jpg', crop_front_small)
            cv2.imwrite(path + '/images/crop-front-small.png', crop_front_small)

            if check_passNum_small==True:
                result['is_Pass_full'] = 'Done'
                result['find_passNum_on_front'] = 'Done'
            else:
                check_passNum = False
                check_passNum_small = False
                result['is_Pass_full'] = 'Pass isn`t full'
                result['find_passNum_on_front'] = 'Couldn`t find PassNum'
    except:
        
        return {'Pass_match': 'None', 'Face_match': 'None', 'Error': 'Couldn`t crop passport on front'}
    
    faces_front = count_faces(img1_p)

    if size=='big':
        faces_count = count_faces(path + '/images/crop-front.jpg')

        if faces_count==1:
            result['Count_faces'] = 1
        else:
            result['Count_faces'] = faces_front

        count_faces_on_front=faces_count
        if (count_faces_on_front == 1) and(check_passNum==True):
            result['find_face_on_front'] = 'Done'
        else:
            result['find_face_on_front'] = f'Count faces - {faces_front}'

        list_numbers = proverka_pass_num(front_pass_num)

        show_numbers = list_numbers

        result['PassNum'] = show_numbers

        if result['find_passNum_on_front'] == 'Done' and result['find_face_on_front'] == 'Done' and result['Find_phone']==False:
            result['RESULT']=True
        else:
            result['RESULT']=False

        return [list_numbers, [result], faces_front, 'Big']
    else:
        faces_count_small = count_faces(path + '/images/crop-front-small.jpg')

        if faces_count==1:
            result['Count_faces'] = 1
        else:
            result['Count_faces'] = faces_front

        count_faces_on_front = max(faces_count, faces_count_small)

        if (count_faces_on_front == 1) and(check_passNum_small==True):
            result['find_face_on_front'] = 'Done'
        else:
            result['find_face_on_front'] = f'Count faces - {faces_front}'
    
    

        list_numbers = proverka_pass_num(front_pass_num_small)

        show_numbers = list_numbers

        result['PassNum'] = show_numbers

        if result['find_passNum_on_front'] == 'Done' and result['find_face_on_front'] == 'Done' and result['Find_phone']==False:
            result['RESULT']=True
        else:
            result['RESULT']=False

        return [list_numbers, [result], faces_front, 'Small']


def true_distance(num):
    if num>100:
        return 100
    elif num<0:
        return 0
    else:
        return num

def selfi_verification(img1_p = 'None'):

    result = {}
    big_face_area(img1_p)
    
    
    try:
        front_img = cv2.imread(img1_p)
        flip_img = cv2.flip(front_img, 1)
        cv2.imwrite(path + '/images/crop-pass-flip.png', flip_img)
        cv2.imwrite(path + '/images/crop-pass-flip.jpg', flip_img)

        if front_img.shape[1]>3000 and front_img.shape[0]>3000:
            h = front_img.shape[1]
            w = front_img.shape[0]
            prop = h/w

            new_w = 1200
            new_h = int(new_w*prop)
            front = cv2.resize(front_img, (new_h, new_w))
            cv2.imwrite(path + '/images/crop-pass-selfi.png', front)
            cv2.imwrite(path + '/images/crop-pass-selfi.jpg', front)


        else:
            front = cv2.imread(img1_p)
            cv2.imwrite(path + '/images/crop-pass-selfi.png', front)
            cv2.imwrite(path + '/images/crop-pass-selfi.jpg', front)
    except:
        return {'Pass_match': 'None', 'Face_match': 'None', 'Error': 'Couldn`t read front image'}
    
    is_flip_pass = is_flip(img1_p)

    if is_flip_pass ==True:

        result['Image_flipped'] = True
        
        result['Find_phone'] = find_phone(img1_p)
        size = ''
        crop_selfie = crop_img(path + '/images/crop-pass-flip.png')
        selfie_pass_num = read_pn(crop_selfie)

        if len(selfie_pass_num)>=5:

            selfie_pass_num = selfie_pass_num
            check_passNum_flip = is_PassNum(selfie_pass_num)
            cv2.imwrite(path + '/images/crop-flip-selfi.jpg', crop_selfie)

            is_watermaker = read_yolo(path + '/images/crop-flip-selfi.jpg')

            if check_passNum_flip == True:
                result['is_Pass_full'] = 'Done'
                result['find_passNum_on_selfie'] = 'Done'
            else:
                check_passNum_flip = False
                check_passNum_flip_small = False
                result['is_Pass_full'] = 'Pass isn`t full'
                result['find_passNum_on_selfie'] = 'Couldn`t find PassNum'

            faces_count = count_faces(path + '/images/crop-flip-selfi.jpg')
            faces_front = count_faces(img1_p)
            result['Count_faces'] = faces_front
            count_faces_on_front = faces_count

            if (count_faces_on_front == 1 and faces_front == 2) and(check_passNum_flip==True):
                
                result['find_face_on_selfie'] = 'Done'
            else:
                result['find_face_on_selfie'] = f'Count faces - {faces_front}'

            list_numbers = proverka_pass_num(selfie_pass_num)
            show_numbers = list_numbers

            result['PassNum'] = show_numbers
            result['Watermark'] = is_watermaker

            if result['find_face_on_selfie'] == 'Done' and result['Find_phone']==False:
                result['RESULT']=True
            else:
                result['RESULT']=False

            ans = [proverka_pass_num(selfie_pass_num), [result], faces_front]

            return ans


        else:
            crop_selfie_small = crop_img2(path + '/images/crop-pass-flip.png')
            selfie_pass_num_small = read_pn2(crop_selfie_small)


            check_passNum_flip_small = is_PassNum(selfie_pass_num_small)

            
            cv2.imwrite(path + '/images/crop-flip-selfi-small.jpg', crop_selfie_small)
            is_watermaker = read_yolo(path + '/images/crop-flip-selfi-small.jpg')

            if check_passNum_flip_small==True:
                result['is_Pass_full'] = 'Done'
                result['find_passNum_on_selfie'] = 'Done'
            else:
                check_passNum_flip = False
                check_passNum_flip_small = False
                result['is_Pass_full'] = 'Pass isn`t full'
                result['find_passNum_on_selfie'] = 'Couldn`t find PassNum'

       
        

            faces_count_small = count_faces(path + '/images/crop-flip-selfi-small.jpg')

            faces_front = count_faces(img1_p)

            result['Count_faces'] = faces_front
            count_faces_on_front = faces_count_small


            if (count_faces_on_front == 1 and faces_front == 2) and(check_passNum_flip_small==True):
                
                result['find_face_on_selfie'] = 'Done'
            else:
                result['find_face_on_selfie'] = f'Count faces - {faces_front}'
        


            list_numbers = proverka_pass_num(selfie_pass_num_small)
            show_numbers = list_numbers

            result['PassNum'] = show_numbers

            result['Watermark'] = is_watermaker

            if result['find_face_on_selfie'] == 'Done' and result['Find_phone']==False:
                result['RESULT']=True
            else:
                result['RESULT']=False

            ans = [proverka_pass_num(selfie_pass_num_small), [result], faces_front]

            return ans
    else:
        result['Image_flipped'] = False

        result['Find_phone'] = find_phone(img1_p)
        try:
            crop_front = crop_img(path + '/images/crop-pass-selfi.png')
            front_pass_num = read_pn(crop_front)
            size = ''
            if len(front_pass_num)>=5:
                size = 'big'

                front_pass_num = front_pass_num
                check_passNum = is_PassNum(front_pass_num)
                cv2.imwrite(path + '/images/crop-selfi.jpg', crop_front)



                if check_passNum == True:
                    result['is_Pass_full'] = 'Done'
                    result['find_passNum_on_selfie'] = 'Done'
                else:
                    check_passNum = False
                    result['is_Pass_full'] = 'Pass isn`t full'
                    result['find_passNum_on_selfie'] = 'Couldn`t find PassNum'
            else:
                size = 'small'

                crop_front_small = crop_img2(path + '/images/crop-pass-selfi.png')
                front_pass_num_small = read_pn2(crop_front_small)
                check_passNum_small = is_PassNum(front_pass_num_small)

                

                cv2.imwrite(path + '/images/crop-selfi-small.jpg', crop_front_small)

                if check_passNum_small == True:
                    result['is_Pass_full'] = 'Done'
                    result['find_passNum_on_selfie'] = 'Done'
                else:
                    check_passNum_small = False
                    result['is_Pass_full'] = 'Pass isn`t full'
                    result['find_passNum_on_selfie'] = 'Couldn`t find PassNum'
        except:
            
            return {'Pass_match': 'None', 'Face_match': 'None', 'Error': 'Couldn`t crop passport on front'}
        
        faces_front = count_faces(img1_p)
        result['Count_faces'] = faces_front
        
        if size == 'big':
        
            faces_count = count_faces(path + '/images/crop-selfi.jpg')
            count_faces_on_front = faces_count
            if (count_faces_on_front == 1 and faces_front == 2) and(check_passNum==True):
                
                result['find_face_on_selfie'] = 'Done'
            else:
                result['find_face_on_selfie'] = f'Count faces - {faces_front}'

            list_numbers = proverka_pass_num(front_pass_num)
            show_numbers = list_numbers

            result['PassNum'] = show_numbers
            is_watermaker = read_yolo(path + '/images/crop-selfi.jpg')
            result['Watermark'] = is_watermaker

            if result['find_face_on_selfie'] == 'Done' and result['Find_phone']==False:
                result['RESULT']=True
            else:
                result['RESULT']=False

            ans = [proverka_pass_num(front_pass_num), [result], size]

            return ans

        else:

            faces_count_small = count_faces(path + '/images/crop-selfi-small.jpg')
            count_faces_on_front = faces_count_small


            if (count_faces_on_front == 1 and faces_front == 2) and(check_passNum_small==True):
                
                result['find_face_on_selfie'] = 'Done'
            else:
                result['find_face_on_selfie'] = f'Count faces - {faces_front}'



            list_numbers = proverka_pass_num(front_pass_num_small)
            show_numbers = list_numbers

            result['PassNum'] = show_numbers

            is_watermaker = read_yolo(path + '/images/crop-selfi-small.jpg')
            result['Watermark'] = is_watermaker
            

            if result['find_face_on_selfie'] == 'Done' and result['Find_phone']==False:
                result['RESULT']=True
            else:
                result['RESULT']=False

            ans = [show_numbers, [result], size]

            return ans

def create_reason(response):
    front_data = response[0][0]
    back_data = response[1][0]
    verif_data = response[2]
    
    if front_data['RESULT'] == False:
        if front_data['is_Pass_full']!='Done':
            return 'Передняя часть паспорта не полностью видна, или паспорт не был найден (1-ое изображение)'    

        if front_data['Count_faces']==0:
            return 'Не видно лица на передней части паспорта (1-ое изображение)'
        
        if front_data['find_passNum_on_front']!='Done':
            return 'Номер паспорта на передней части не был правильно распознан (1-ое изображение)'
        
        if front_data['Find_phone']!=False:
            return 'На фотографии паспорта присутствует телефон или другое устройство'
        
        

    if back_data['RESULT']==False:
        if back_data['is_Pass_full']!='Done':
            return 'Паспорт на селфи не полностью виден, или паспорт не был найден'
        
        if back_data['find_passNum_on_selfie']!='Done':
            return 'Номер паспорта на селфи не был правильно распознан'

        if back_data['Count_faces']!=2:
            return 'Лицо человека с паспортом не было правильно обнаружена'

    if verif_data['Verified']==False:
        
        if verif_data['Pass_match']==False:
            return 'Не удалось удоствериться в идентичности паспортов на изображениях'

        if verif_data['Face_match']==False:
            return 'Не удалось удоствериться что паспорт принадлежит вам (Несхожесть в лицах)'

        if verif_data['is_fake']=='Fake':
            return 'Не удалось распознать лицо на селфи, убедитесь что освещение является хорошим'

        if verif_data['Selfi_Original_image']!=True:
            return 'Фото - селфи не является оригинальным или повреждено. Пожалуйста, переснимите с помощью камеры вашего телефона'
        
        if verif_data['Front_Original_image']!=True:
            return 'Фото - паспорта не является оригинальным или повреждено. Пожалуйста, переснимите с помощью камеры вашего телефона'
        
        
    return 'Успешно пройдено!'

def verification(front, selfi,exif,exif_check_front,  user_thresh='optimistic'):
	now = time.time()
	result = {}
	selfi_ans = selfi_verification(selfi)

	front_ans = front_verification(front)


	try:
		front = front_ans[0]
	except:
		front = None

	try:
		selfi = selfi_ans[0]
	except:
		selfi = None
	result['Front_Original_image'] = exif_check_front
	result['Selfi_Original_image'] = exif
	try:
		if front==selfi and front_ans[1][0]['RESULT']==True and selfi_ans[1][0]['RESULT']==True:
			result['Pass_match'] = True
		else:
			result['Pass_match']=False
	except:
		result['Pass_match'] = None

	im = path + f'/images/crop-selfi-bigface.jpg'
	is_fake_img = is_fake(im)
    
	try:
		big_face = true_distance(insight_model(path + '/images/crop-pass.jpg', path + '/images/crop-selfi-bigface.jpg'))
	except:
		big_face = 0

	if user_thresh=='optimistic':
		threshold = 30
	else:
		threshold = 40

	try:
		if big_face>=threshold:
			result['Face_match'] = True
		else:
			result['Face_match'] = False
	except:
		result['Face_match'] = None

	try:
		result['is_fake'] = is_fake_img[0]  
		result['fake_score'] = float(round(is_fake_img[1],2))
	except:
		result['is_fake'] = None
		result['fake_score'] = 0

	try:
		result['BigFace'] = float(big_face)
	except:
		result['BigFace'] = 0
    

	result['Threshold'] = user_thresh
	try:
		if big_face>=threshold and result['Pass_match']==True and is_fake_img[0]=='Real' and result['Front_Original_image']==True and result['Selfi_Original_image']==True:
			result['Verified'] = True
		else:
			result['Verified'] = False
	except:
		result['Verified'] = None
	              
               
	fin = time.time()

    

	try:

		front_res = front_ans[1]
	except:
		front_res = {
               'Find_phone': False,
               'is_Pass_full': False,
               'find_passNum_on_front': False,
               'Count_faces': 0,
               'find_face_on_front': None,
               'PassNum': None,
               'RESULT': False
			}
	try:
		selfi_res = selfi_ans[1]
	except:
		selfi_res = {
               'Image_flipped': False,
               'Find_phone': False,
               'is_Pass_full': False,
               'find_passNum_on_front': False,
               'Count_faces': 0,
               'find_face_on_selfie': False,
               'PassNum': None,
               'Watermark': False,
               'RESULT': False
        }
	try:
		shutil.rmtree('runs/')
	except:
		pass
	try:
		reason = create_reason([front_res,selfi_res,result])
	except Exception as ex:
		print(ex.__str__())
		reason = 'Неизвестная ошибка'
	return [front_res,selfi_res,result, reason]


from flask import Flask, request, Response
import jsonpickle

app = Flask(__name__)

def get_encoded_id(text = 'None'):
  sha_encoder = hashlib.sha256()
  sha_encoder.update(text.encode('utf-8'))
  encoding = sha_encoder.hexdigest()
  id = 0
  for i in range(len(encoding)):
    coef = 1+i/10
    id = id + ord(encoding[i])*coef
  return int(id)

def sum_img_pixels( img_path='None',image='None'):
  if image!='None':
    img = image
  else:
    img = cv2.imread(img_path)
  
  summa = 0
  
  for i in range(len(img[0])):
    line = img[0][i]

    coef = 1+i/10
    for j in range(len(line)):
      summa = summa + (line[j]*coef)
  return int(summa)


@app.route('/front_recognition', methods=['POST'])
def front():
    rand_front = random.randint(1, 100000)

    front_img = request.files['front']

    file_format = 'jpg'
    if front_img.filename.lower().endswith('.png'):
        file_format = 'png'

    img = Image.open(front_img.stream)

    file_size = round((int(len(img.fp.read()))/1024)/1024, 2)

    filename = path + f'/images/request_api/front/{rand_front}.{file_format}'
    img.save(filename)

    get_req = requests.get('http://91.222.237.176:2354/humo/exe_front')

    if str(get_req.text) == str(True):
        accept_header = front_img.filename
        format_img = accept_header.split('.')[-1]

        t1 = time.time()
        front_res = read_passport(filename)

        t2 = time.time()

        find_pass = True
        if len(front_res['Name_rus'])>=3 and len(front_res['Date_of_birth'])>=3 and len(front_res['Pass_num'])>=3:
            find_pass = True
        else:
            find_pass = False
        data = {
            'name': 'humo',
            'img_size': str([cv2.imread(filename).shape[0], cv2.imread(filename).shape[1]]),
            'weight_img': file_size,
            'special_id': sum_img_pixels(filename),
            'pass_num': get_encoded_id(front_res['Pass_num']),
            'format_img': str(format_img),
            'find_pass': str(find_pass),
            'time': str(round(t2-t1, 2)),
            'date': str(datetime.now())
        }
        post_req = requests.post('http://91.222.237.176:2354/humo/exe_front', data)


        response = {'Front_result': front_res}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype='application/json')
    else:
        response_pickled = jsonpickle.encode({'Couldnt request to server'})
        return Response(response=response_pickled, status=200, mimetype='application/json')




@app.route('/back_recognition', methods=['POST'])
def back():
    rand_front = random.randint(1, 100000)

    front_img = request.files['back']

    file_format = 'jpg'
    if front_img.filename.lower().endswith('.png'):
        file_format = 'png'

    img = Image.open(front_img.stream)

    file_size = round((int(len(img.fp.read()))/1024)/1024, 2)

    filename = path + f'/images/request_api/back/{rand_front}.{file_format}'
    img.save(filename)


    get_req = requests.get('http://91.222.237.176:2354/humo/exe_back')

    if str(get_req.text) == str(True):
        accept_header = front_img.filename
        format_img = accept_header.split('.')[-1]

        t1 = time.time()
        front_res = levenshtein(filename)

        t2 = time.time()

        find_pass = True
        if len(front_res['Address'])>=3 and len(front_res['Marital_status'])>=3 and len(front_res['Tax_payer_ID_number'])>=3:
            find_pass = True
        else:
            find_pass = False
        data = {
            'name': 'humo',
            'img_size': str([cv2.imread(filename).shape[0], cv2.imread(filename).shape[1]]),
            'weight_img': file_size,
            'special_id': sum_img_pixels(filename),
            'pass_num': 'None',
            'format_img': str(format_img),
            'find_pass': str(find_pass),
            'time': str(round(t2-t1, 2)),
            'date': str(datetime.now())
        }
        post_req = requests.post('http://91.222.237.176:2354/humo/exe_back', data)


        response = {'Back_result': front_res}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype='application/json')
    else:
        response_pickled = jsonpickle.encode({'Couldnt request to server'})
        return Response(response=response_pickled, status=200, mimetype='application/json')





@app.route('/verification', methods=['POST'])
def verify():
    rand_front = random.randint(1, 100000)

    selfi_img = request.files['selfi']


    front_img = request.files['front_pass']
    img = Image.open(front_img.stream)

    weight_front = round((int(len(img.fp.read()))/1024)/1024, 2)
    exif_front = check_exif(img)
    img3 = img.convert('RGB')

    file_format = 'jpg'
    if front_img.filename.lower().endswith('.png'):
        file_format = 'png'

    img3.save(path + f'/images/request_api/verif/front-{rand_front}.{file_format}')


    file_back = request.files['selfi']

    file_format1 = 'jpg'
    if file_back.filename.lower().endswith('.png'):
        file_format1 = 'png'
    img1 = Image.open(file_back.stream)

    weight_selfi = round((int(len(img1.fp.read()))/1024)/1024, 2)
    exif_check = check_exif(img1)
    img2 = img1.convert('RGB')
    img2.save(path + f'/images/request_api/verif/selfi-{rand_front}.{file_format1}')

    get_req = requests.get('http://91.222.237.176:2354/humo/exe_verify')

    filename_front = path + f'/images/request_api/verif/front-{rand_front}.{file_format}'
    filename_selfi = path + f'/images/request_api/verif/selfi-{rand_front}.{file_format1}'

    if str(get_req.text) == str(True):
        import time
        
        size_front = [cv2.imread(filename_front).shape[0], cv2.imread(filename_front).shape[1]]
        size_selfi = [cv2.imread(filename_selfi).shape[0], cv2.imread(filename_selfi).shape[1]]

        accept_header = front_img.filename
        format_img_front = accept_header.split('.')[-1]

        accept_header1 = selfi_img.filename
        format_img_selfi = accept_header1.split('.')[-1]



        t1 = time.time()
        verific = verification(filename_front, filename_selfi, exif=exif_check, exif_check_front = exif_front)

        t2 = time.time()

        is_pass_front = True
        if verific[0][0]['is_Pass_full']=='Done':
            is_pass_front=True 
        else:
            is_pass_front = False


        is_pass_selfi = True
        if verific[1][0]['is_Pass_full']=='Done':
            is_pass_selfi=True 
        else:
            is_pass_selfi = False

        pass_match = verific[2]['Pass_match']
        face_match = verific[2]['Face_match']

        dist = verific[2]['BigFace']

        time = round(t2-t1)
        front_passNum = verific[0][0]['PassNum']
        selfi_passNum = verific[1][0]['PassNum']
        is_fake = verific[2]['is_fake']

        datetime_now = datetime.now()


        data = {
            'name': 'humo',
            'front_img_size': str(size_front),
            'selfi_img_size': str(size_selfi),

            'format_front': str(format_img_front),
            'format_selfi': str(format_img_selfi),

            'weight_front_img': weight_front,
            'weight_selfi_img': weight_selfi,

            'front_findPass': str(is_pass_front),
            'selfi_findPass': str(is_pass_selfi),

            'front_passNum': get_encoded_id(str(front_passNum)),
            'selfi_passNum': get_encoded_id(str(selfi_passNum)),

            'pass_match': str(pass_match),
            'face_match': str(face_match),
            'distance': str(dist),

            'front_original': verific[2]['Front_Original_image'],
            'selfi_original': verific[2]['Selfi_Original_image'],

            'front_len_passnum': len(verific[0][0]['PassNum']),
            'selfi_len_passnum': len(verific[1][0]['PassNum']),

            'front_special_id': sum_img_pixels(filename_front),
            'selfi_special_id': sum_img_pixels(filename_selfi),

            'explanation_of_reject': verific[3],


            'is_fake': str(is_fake),
            'time': str(time),
            'date': str(datetime_now)
        }
        post_req = requests.post('http://91.222.237.176:2354/humo/exe_verify', data)


        response = {'Verification': verific}
        response_pickled = jsonpickle.encode(response)
        return Response(response=response_pickled, status=200, mimetype='application/json')
    else:
        response_pickled = jsonpickle.encode({'Couldnt request to server'})
        return Response(response=response_pickled, status=200, mimetype='application/json')

app.run(host='0.0.0.0')
