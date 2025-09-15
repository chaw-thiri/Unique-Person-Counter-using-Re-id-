# Person Re-Identification and Counting with YOLO11n + BoT-SORT
This project integrate human detection &amp; tracking models to detect the accurate number of people present in a scene. One big struggle of object trackers is that the trackers cannot remember whether an object has appeared before. This is solved using BoT-SORT's ReID procedure. This ensures the same person retains he/her ID when they reappear.

This project demonstrates **real-time person detection, tracking, and re-identification (ReID)** using YOLO and BoT-SORT tracker.  

Unlike simple trackers that lose track of people during occlusion or camera movement, **ReID ensures that the same person retains their ID even after disappearing and reappearing**.  

The system also counts the **number of unique people** who appear in the video.

---

## ✨ Features

- **Person detection** with YOLO11 models  
- **Re-identification (ReID)** using BoT-SORT → keeps the same ID after occlusion  
- **Unique person counting** → tracks how many distinct people appeared throughout the video  
- **Configurable frame-skipping** for faster processing (`process_every_n`)  
- **Export annotated video** with bounding boxes, IDs, and person count overlay  

---

## 📊 Example Output

- Bounding boxes around detected people  
- Each person assigned a unique **persistent ID**  
- Total unique person count displayed on screen  
- Final summary of total unique people at the end  

---

## 🎬 Demo

[![Watch the demo](https://github.com/user-attachments/assets/8140f0c1-45e4-4bfc-89b7-c91220f771bc)](https://github.com/user-attachments/assets/8140f0c1-45e4-4bfc-89b7-c91220f771bc)


  

**Explanation of the Demo:**  
In the above video, the person with **ID 5** got temporarily **occluded by ID 1**. However, as soon as ID 5 reappeared, the tracker correctly **regained its original ID (5)**, proving that **ReID successfully prevents ID switches after occlusion**.  
[![Download another demo](https://github.com/chaw-thiri/Unique-Person-Counter-using-Re-id-/blob/main/thumbnail.png)](https://github.com/chaw-thiri/Unique-Person-Counter-using-Re-id-/blob/main/t_homeplus1.mp4)

---

## 🛠️ Use Cases

This system can be adapted for a wide range of real-world applications:  

1. **Crowd Analytics** – Count the number of unique visitors in shopping malls, stadiums, or public spaces.  
2. **Security & Surveillance** – Track individuals across multiple cameras, even after temporary occlusion.  
3. **Event Monitoring** – Estimate crowd size at concerts, rallies, or festivals.  
4. **Smart Retail** – Understand customer foot traffic patterns without duplicate counts.  
5. **Workplace Safety** – Monitor restricted zones to ensure only authorized personnel are present.  

---

## 🚀 How It Works

1. **YOLO Detection** – Detects persons in each frame (`class=0`).  
2. **BoT-SORT Tracking** – Assigns IDs and tracks movements.  
3. **ReID Embeddings** – Ensures consistent IDs across occlusions.  
4. **Unique ID Tracking** – Counts the number of distinct IDs that appeared.  
5. **Video Export** – Saves an annotated video with IDs and statistics.  

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/person-reid-counter.git
cd person-reid-counter
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python main.py --video path/to/your/video.mp4
```

---

## ✅ Example Result

```
👥 Total unique persons detected: 12
✅ Output video saved as: output_video.mp4
