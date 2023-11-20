import cv2
import sys
from ultralytics import YOLO

# Load model kustom yang telah dilatih
model = YOLO(r'models\custom.pt')  # Ganti 'custom_model.pt' dengan nama file model Anda

def detect_objects(frame):
    # Lakukan prediksi menggunakan model
    results = model.predict(frame)

    # Tampilkan hasil deteksi objek pada frame video
    class_id = ['Gunting', 'Kapak', 'Linggis', 'Palu', 'Pisau', 'Pistol']
    for r in results:
        for box in r.boxes:
            b = box.xyxy[0]
            c = int(box.cls)
            label = f"Class: {class_id[c]}"
            confidence = f"Confidence: {float(box.conf):.2f}"
            if confidence >= 0.5:
                # Tambahkan logika untuk menandai atau menggambar bounding box dan label pada frame di sini
                cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(b[0]), int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, confidence, (int(b[0]), int(b[1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return frame

def read(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lakukan deteksi objek pada setiap frame
        detected_frame = detect_objects(frame)

        # Tampilkan frame video dengan hasil deteksi objek
        cv2.imshow('Object Detection in Video', detected_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python detectpicture.py 'nama imagenya'")
    else:
        file_path = sys.argv[1]
        read(file_path)

if __name__ == "__main__":
    main()