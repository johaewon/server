import torch
from deformable_detr import DeformableDetrFeatureExtractor, DeformableDetrConfig
from PIL import Image
import requests
from egtr import DetrForSceneGraphGeneration  # egtr.py 파일에 있는 DetrForSceneGraphGeneration 클래스
from transformers import DeformableDetrFeatureExtractor, DeformableDetrConfig

#################################
# 사전 학습된 모델

egtr_ckpt_path = "egtr-main\trained_model\egtr\epoch=03-validation_loss=1.71.ckpt"
egtr_config = 'egtr-main\trained_model\egtr\egtr_config.json'

# 설정 파일 로드
config = DeformableDetrConfig.from_pretrained(egtr_config)

# DetrForSceneGraphGeneration 모델 초기화하고 체크포인트에서 가중치 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DetrForSceneGraphGeneration.load_from_checkpoint(egtr_ckpt_path)
checkpoint = torch.load(egtr_ckpt_path, map_location = torch.device(device))
model.load_state_dict(checkpoint['state_dict'])


# feature extractor 로드
feature_extractor = DeformableDetrFeatureExtractor.from_pretrained('egtr-main\trained_model\deformable_detr\pytorch_model.bin')


# 이미지 로드
image_url = url = "./sample_data/image1.jpg"
image = Image.open(requests.get(image_url, stream = True).raw)

# 전처리 진행
inputs = feature_extractor(images = image, return_tensors = "pt")


model.eval()
with torch.no_grad():
    outputs = model(**inputs)



# 예측 결과 처리
logits = outputs.logits
pred_boxes = outputs.pred_boxes
pred_rel = outputs.pred_rel
pred_connectivity = outputs.pred_connectivity


# 예측된 바운딩 박스 이미지에 그리기
def visualize_results(image, logits, pred_boxes, pred_rel, pred_connectivity):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Convert logits to probabilities
    prob = logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), 100, dim=1)
    scores = topk_values
    labels = topk_indexes % logits.shape[2]
    
    # Convert boxes from [0, 1] to absolute coordinates
    img_h, img_w = image.size
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
    boxes = pred_boxes * scale_fct

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)

    # Draw bounding boxes
    for box, label, score in zip(boxes[0], labels[0], scores[0]):
        if score > 0.5:  # Threshold to filter out low-confidence boxes
            x, y, w, h = box
            rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f'{label.item()}:{score.item():.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

    plt.show()


visualize_results(image, logits, pred_boxes, pred_rel, pred_connectivity)