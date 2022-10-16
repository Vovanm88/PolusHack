import os
import json
import numpy as np
import cv2
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model
def upload_model(mpath="/content/model4.pt"):
    model = get_model_instance_segmentation()
    model.load_state_dict(torch.load(mpath))
    print("Модель успешно загружена")
    return model


class ImagePipeline:
    def __init__(self, path="/content/model4.pt", device):
        self.model = upload_model(path).to(device)
        self.dev = device

    def __call__(self, input_image, threshold=1, value=1):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transform = transforms.ToTensor()
        img = transform(image).unsqueeze(0)
        self.model.eval()
        im2=img.to(self.dev)
        output = self.model(im2)
        data_out = {"stones": [], ""}

        for j, i in enumerate(output[0]['boxes'].cpu().detach().numpy()):
            x = abs(int(i[0]-i[2]))
            y = abs(int(i[1]-i[3]))
            x0 = int(i[0])
            y0 = int(i[1])
            x1 = int(i[2])
            y1 = int(i[3])

            
            stocorr = (y0-487)*1340/510
            stomins = stocorr*1600/500
            stones = []

            if isInCircle(i[0], i[1], 1920) and isInCircle(i[2], i[3], 1920) and output[0]['scores'].cpu().detach().numpy()[j]>0.3:
                if stomins>threshold:
                    cv2.rectangle(image, [int(i[0]), int(i[1])], [int(i[2]), int(i[3])], (255, 0, 0), 4)
                else:
                    cv2.rectangle(image, [int(i[0]), int(i[1])], [int(i[2]), int(i[3])], (0, 255, 0), 2)
                stones.append(stomins)
        data_out["stones"] = stones
        output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return output_image, data_out

pipe = ImagePipeline(path_for_model)

img, data_json = pipe(img, polzunok)