import cv2
from ultralytics import YOLO

# Load model kustom yang telah dilatih
model = YOLO('D:\StartUp\dataset\models\yolov8n.pt')  # Ganti 'custom_model.pt' dengan nama file model Anda

# Baca gambar untuk deteksi objek
img = cv2.imread('D:\StartUp\dataset\R.jpg')
img_resized = cv2.resize(img, (640, 480))  # Resize gambar menjadi 640x480

# Lakukan prediksi menggunakan model
results = model.predict(img_resized)

# Ukuran gambar yang diubah
new_height, new_width = img_resized.shape[:2]

# Tampilkan hasil deteksi objek pada gambar yang diubah ukurannya
for r in results:
    for box in r.boxes:
        b = box.xyxy[0]
        c = int(box.cls)
        label = f"Class: {c} - Confidence: {float(box.conf):.2f}"
        print(label)
        print(b)

        # Ubah koordinat bounding box sesuai dengan skala gambar yang diubah
        x1 = int(b[0] * new_width)
        y1 = int(b[1] * new_height)
        x2 = int(b[2] * new_width)
        y2 = int(b[3] * new_height)

        # Tambahkan logika untuk menandai atau menggambar bounding box dan label pada gambar di sini
        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Tampilkan gambar dengan hasil deteksi
cv2.imshow("YOLO Object Detection", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
