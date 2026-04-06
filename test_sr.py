
import cv2
import os

def process_tessize():
    image_path = r'C:\VS projects\Sem 7\SuperResolution\tessize.png'
    model_path = r'C:\VS projects\Sem 7\EDSR_x4.pb'
    output_path = r'C:\VS projects\Sem 7\SuperResolution\upscaled_tessize.png'
    
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print("Initializing Super Resolution...")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    
    print("Reading model...")
    sr.readModel(model_path)
    
    print("Setting model...")
    sr.setModel('edsr', 4)
    
    print("Reading image...")
    img = cv2.imread(image_path)
    
    print("Upsampling...")
    upscaled = sr.upsample(img)
    
    print("Saving result...")
    cv2.imwrite(output_path, upscaled)
    print(f"Processed image saved to {output_path}")

if __name__ == "__main__":
    process_tessize()
