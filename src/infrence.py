import cv2
import torch
from ultralytics import YOLO
from torchvision import models, transforms
import numpy as np
from torch.nn.functional import cosine_similarity
from PIL import Image

# Load Models
yolo_model = YOLO('models/detection_model/yolov8_person.pt')

attr_model = models.resnet50()
num_attributes = 5
attr_model.fc = torch.nn.Linear(attr_model.fc.in_features, num_attributes)
attr_model.load_state_dict(torch.load('models/attribute_model/attribute_model.pth', map_location='cpu'))
attr_model.eval()

reid_model = models.resnet50()
reid_model.fc = torch.nn.Identity()
reid_model.load_state_dict(torch.load('models/reid_model/reid_model.pth', map_location='cpu'))
reid_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attr_model.to(device)
reid_model.to(device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

known_embeddings = []
known_ids = []
id_counter = 0

# Camera Capture
camera_url = "rtsp://your-camera-ip"
cap = cv2.VideoCapture(camera_url)

def predict_attributes(crop_img):
    img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.sigmoid(attr_model(input_tensor)).cpu().numpy()[0]
    return preds

def extract_embedding(crop_img):
    img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(input_tensor).cpu()
    return embedding

def match_embedding(embedding, threshold=0.7):
    global known_embeddings, known_ids, id_counter
    if len(known_embeddings) == 0:
        known_embeddings.append(embedding)
        known_ids.append(id_counter)
        id_counter += 1
        return known_ids[-1]
    sims = [cosine_similarity(embedding, e).item() for e in known_embeddings]
    max_sim = max(sims)
    if max_sim > threshold:
        return known_ids[sims.index(max_sim)]
    else:
        known_embeddings.append(embedding)
        known_ids.append(id_counter)
        id_counter += 1
        return known_ids[-1]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame")
        break

    results = yolo_model(frame)
    boxes = results[0].boxes
    person_boxes = [b for b in boxes if int(b.cls[0])==0]

    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        attr_preds = predict_attributes(crop)
        embedding = extract_embedding(crop)
        person_id = match_embedding(embedding)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        attr_str = ','.join([f"{i:.2f}" for i in attr_preds])
        cv2.putText(frame, f"ID:{person_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("IP Camera Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
