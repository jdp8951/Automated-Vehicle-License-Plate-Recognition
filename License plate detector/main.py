from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # build a new model from scratch

# Use the model
results = model.train(data="D:\AML Project\License plate detector\Dataset\data.yaml", epochs=10)  # train the model