from ultralytics import YOLO
import torch

def main():

    model = YOLO("yolo11s.pt") 

    results = model.train(data=r"C:\Users\user\Desktop\weight_recognition\data.yaml",
                            epochs=100, 
                            imgsz=640, 
                            batch=8,
                            # hsv_h=0.0,
                            # hsv_s=0.0,
                            # hsv_v=0.0,
                            # translate=0.0,
                            # scale=0.0,
                            # fliplr=0.0,
                            # mosaic=0.0,
                            # erasing=0.0,
                            # auto_augment=None,
                            )

if __name__ == '__main__':
    with torch.no_grad():
        torch.cuda.empty_cache()
    main()  
