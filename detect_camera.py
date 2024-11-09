import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator


MODEL_PATH = 'runs/train/exp/weights/last.pt' 


device = select_device('')
model = DetectMultiBackend(MODEL_PATH, device=device)
stride, names, pt = model.stride, model.names, model.pt
img_size = 640

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  
    if len(img.shape) == 3:
        img = img[None]  


    pred = model(img)
    pred = non_max_suppression(pred, 0.25, 0.45) 

    for det in pred:
        im0 = frame.copy()
        annotator = Annotator(im0, line_width=3, example=str(names))

        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=(255, 0, 0)) 

        cv2.imshow('Anjay Jadi', im0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
