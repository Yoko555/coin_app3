#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch
from data_augment import ValTransform
from coco_classes import COCO_CLASSES
from build import get_exp
from model_utils import fuse_model, get_model_info
from boxes import postprocess
from visualize import vis
import tempfile

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def make_empty_parser():
    return argparse.ArgumentParser()


def make_parser():
    logger.info(f"demo.py : IN make_parser 1")
    parser = argparse.ArgumentParser("YOLOX Demo!")
    logger.info(f"demo.py : IN make_parser 2")
    parser.add_argument(
        "demo", default="image", nargs="?", help="demo type, eg. image, video and webcam"
    )
    logger.info(f"demo.py : IN make_parser 3")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./datafolder/saved_image.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    logger.info(f"demo.py : IN make_parser 4")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

        logger.info("log predictor cls_names {}".format(cls_names))

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        #logger.info(f"demo.py : img = {img}")
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        print(f"log inference outputs{outputs}")
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        logger.info("result_image {}".format(self.cls_names))
        #print(f"log visual vis_res : {vis_res}")
        return vis_res


#def image_demo(predictor, vis_folder, path, current_time, save_result):
def image_demo(predictor, path, current_time, save_result):
    logger.info("image_demo save_result1 : {}".format(save_result))
    logger.info(f"demo.py : path = {path}")
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        logger.info("demo.py image_demo image_name {}".format(image_name))
        outputs, img_info = predictor.inference(image_name)
        logger.info("demo.py image_demo after predictor.inference")
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        logger.info("demo.py image_demo after predictor.visual")
        #logger.info(f"demo.py : result_image = {result_image}")

#####
        # YOLOXの推論結果からクラスIDを抽出して表示
        class_counts = {0:0}
        for output in outputs:
            # outputは(batch_size, 6)の形状を持つテンソルで、各行が物体の情報を表しています
            for detection in output:
                class_id = int(detection[6])  # 7番目の要素がクラスIDを表しています
                #print(f"Detected Class ID: {class_id}")
                # クラスIDごとにカウントを増やす
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

        logger.info("demo.py image_demo after predictor.visual")
        AMOUNT_LIST = [item.replace("JPY","") for item in COCO_CLASSES]

        # クラスごとのカウントを表示
        total_amount=0
        for class_id, count in class_counts.items():
            total_amount += int(AMOUNT_LIST[class_id]) * count
            print(f"Class ID {class_id}={AMOUNT_LIST[class_id]}円: {count}個 合計{total_amount}円")

        logger.info("total amount {}".format(total_amount))
#####

        if save_result:
#            save_folder = vis_folder
#            save_folder = os.path.join(
#                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#            )
#            logger.info(f"save_folder: {save_folder}")
#            os.makedirs(save_folder, exist_ok=True)
            with tempfile.NamedTemporaryFile(delete=False,suffix=".png") as temp_file_result:
                cv2.imwrite(temp_file_result.name, result_image) 
#                outputs.save(temp_file_result)
                temp_file_result_path = temp_file_result.name
            logger.info(f"temp_file_result_path: {temp_file_result_path}")

#            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
#            logger.info("Saving detection result in {}".format(save_file_name))
#            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
    
#    logger.info("image_demo save_file_name : {}".format(save_file_name))
#    return total_amount, save_file_name
    return total_amount, temp_file_result_path


#def imageflow_demo(predictor, vis_folder, current_time, args):
#    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
#    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
#    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
#    fps = cap.get(cv2.CAP_PROP_FPS)
#    if args.save_result:
#        save_folder = os.path.join(
#            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#        )
#        os.makedirs(save_folder, exist_ok=True)
#        if args.demo == "video":
#            save_path = os.path.join(save_folder, os.path.basename(args.path))
#        else:
#            save_path = os.path.join(save_folder, "camera.mp4")
#        logger.info(f"video save_path is {save_path}")
#        vid_writer = cv2.VideoWriter(
#            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
#        )
#    while True:
#        ret_val, frame = cap.read()
#        if ret_val:
#            outputs, img_info = predictor.inference(frame)
#            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
#            if args.save_result:
#                vid_writer.write(result_frame)
#            else:
#                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
#                cv2.imshow("yolox", result_frame)
#            ch = cv2.waitKey(1)
#            if ch == 27 or ch == ord("q") or ch == ord("Q"):
#                break
#        else:
#            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    logger.info(f"demo main exp.output_dir: {exp.output_dir}")
    logger.info(f"demo main args.experiment_name: {args.experiment_name}")
#    os.makedirs(file_name, exist_ok=True)

#    vis_folder = None
    logger.info(f"demo main args.save_result: {args.save_result}")
#    if args.save_result:
#        vis_folder = os.path.join(file_name, "vis_res")
#        os.makedirs(vis_folder, exist_ok=True)
#        logger.info(f"vis_folder: {vis_folder}")

    if args.trt:
#        args.device = "gpu"
        args.device = "cpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "image":

        logger.info(f"args.save_result2 : img = {args.save_result}")
#        total_amount = image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
        total_amount = image_demo(predictor,  args.path, current_time, args.save_result)
#    elif args.demo == "video" or args.demo == "webcam":
#        total_amount = imageflow_demo(predictor, vis_folder, current_time, args)
    return total_amount

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
