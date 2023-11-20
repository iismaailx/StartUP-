import sys
import cv2
from ultralytics import YOLO

# Load model kustom yang telah dilatih
model = YOLO(r'models\custom.pt')  # Ganti 'custom_model.pt' dengan nama file model Anda

def detect(filepath):
    # Baca gambar untuk deteksi objek
    img = cv2.imread(filepath)
    img = cv2.resize(img, (640, 640))
    # Lakukan prediksi menggunakan model
    results = model.predict(img)

    # Tampilkan hasil deteksi objek pada gambar
    class_id = ['Gunting', 'Kapak', 'Linggis', 'Palu', 'Pisau', 'Pistol']
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0]
            c = int(box.cls)
            label = f"Class: {class_id[c]}"
            confident = f"Confident: {float(box.conf):.2f}"
            print(label)
            print(b)
            if confident >= 0.5:
                # Tambahkan logika untuk menandai atau menggambar bounding box dan label pada gambar di sini
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(img, label, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(img, confident, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Tampilkan gambar dengan hasil deteksi
    cv2.imshow("The Explores Object Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python detectpicture.py 'nama imagenya'")
    else:
        file_path = sys.argv[1]
        detect(file_path)

if __name__ == "__main__":
    main()