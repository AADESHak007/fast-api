from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import cv2
import mediapipe as mp
import numpy as np
import os

app = FastAPI()

# Enable CORS (allow all origins or restrict to your frontend domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/mask-nose")
async def mask_nose(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp_input.write(await file.read())
    temp_input.close()

    image = cv2.imread(temp_input.name)
    if image is None:
        return {"error": "Failed to load image"}

    h, w, _ = image.shape

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        try:
            # Nose tip
            nose_tip = face_landmarks.landmark[1]
            cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)

            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]
            lx, ly = int(left_cheek.x * w), int(left_cheek.y * h)
            rx, ry = int(right_cheek.x * w), int(right_cheek.y * h)
            face_width = np.linalg.norm([rx - lx, ry - ly])
            radius = int(face_width * 0.20)

            # Draw a white filled circle at the nose tip
            cv2.circle(mask, (cx, cy), radius, 255, -1)

            # Draw triangle above the circle for nose bridge
            base_left = (cx - radius // 2, cy - radius)
            base_right = (cx + radius // 2, cy - radius)
            apex = (cx, cy - int(radius * 2.2))
            triangle_cnt = np.array([base_left, base_right, apex])
            cv2.drawContours(mask, [triangle_cnt], 0, 255, -1)

        except Exception as e:
            return {"error": f"Face detection fallback failed: {str(e)}"}

    else:
        return {"error": "No face detected"}

    # Save output mask to a temporary file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(temp_output.name, mask)

    # Return the file
    return FileResponse(temp_output.name, media_type="image/png")