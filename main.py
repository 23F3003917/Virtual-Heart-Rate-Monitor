from pyVHR.analysis.pipeline import Pipeline
import cv2
import numpy as np
import pandas as pd
import os

# PARAMETERS
wsize = 5
roi_approach = 'holistic'
bpm_est = 'clustering'
method = 'cpu_CHROM'
video_path = "/content/drive/MyDrive/ANI1/Trial/pyVHR/10a.mp4"
output_video_path = "/Users/anirudhsinghtomar/Desktop/10a_with_BPM.mp4"


# Load video and get FPS and resolution
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

# Run pyVHR pipeline
pipe = Pipeline()
bvps, timesES, bpmES = pipe.run_on_video(
    video_path,
    winsize=wsize,
    roi_method='convexhull',
    roi_approach=roi_approach,
    method=method,
    estimate=bpm_est,
    patch_size=0,
    RGB_LOW_HIGH_TH=(5, 230),
    Skin_LOW_HIGH_TH=(5, 230),
    pre_filt=True,
    post_filt=True,
    cuda=False,
    verb=True
)

# Calculate average BPM
est_bpm = np.mean(bpmES)
print(f"✅ BPM: {est_bpm:.2f}")

# Prepare video reader and writer
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Overlay average BPM text
    bpm_text = f"Est BPM: {est_bpm:.1f}"
    cv2.putText(frame, bpm_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # Write the frame
    out.write(frame)

cap.release()
out.release()

print(f"🎥 Output video with BPM saved to: {output_video_path}")

# Compute average BPM and save to CSV
est_bpm = np.mean(bpmES)

df_avg = pd.DataFrame({
    'Est BPM': [est_bpm]
})
df_avg.to_csv("10a_BPM.csv", index=False)
print(" BPM saved to csv")
