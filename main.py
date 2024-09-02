from ultralytics import YOLO
model = YOLO("yolov8m-oiv7.pt")
results = model.train(data="D:/YOLO/fish/data.yaml", epochs=5, imgsz=640, warmup_epochs=0, device="cpu")

# results = model.train(
#     data="D:/YOLO/fish/data.yaml",  # 指定數據集配置文件的路徑
#     epochs=5,                 # 訓練的輪次設置為100
#     imgsz=640,                  # 輸入圖像的大小設置為640
#     # batch=128,                  # 批次大小設置為128，增加批次大小以提高訓練效率
#     # workers=128,                # 數據加載器的工作數設置為128，增加數據加載效率
#     # amp=True,                   # 啟用自動混合精度（AMP）以加速訓練並節省內存
#     # cos_lr=True,                 # 使用餘弦退火學習率調度器
#     warmup_epochs=0
# )



# 加載訓練好的模型
model = YOLO("runs/detect/train15/weights/best.pt")
results = model.predict(source="D:/YOLO/image/00.jpg",save=True)  # predict on an image
# 測試視頻並逐幀處理
results = model.predict(source="D:/YOLO/youtubevideo.mp4", stream=True, save=True)

# 逐幀處理結果
for r in results:
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs
    print("Boxes:", boxes)
    print("Masks:", masks)
    print("Probs:", probs)