import cv2
import numpy as np
import mediapipe as mp             ## MediaPipe Face Mesh to extract forehead + cheek ROIs (most signal-rich regions)
from dataclasses import dataclass  ## Clean way to store output
from typing import Optional

## Container for output data
@dataclass
class FaceROI:
    """Extracted face region data."""
    forehead_rgb:  np.ndarray          # H×W×3 forehead crop
    left_cheek_rgb: np.ndarray         # H×W×3 left cheek crop
    right_cheek_rgb: np.ndarray        # H×W×3 right cheek crop
    combined_rgb:  np.ndarray          # stacked mean region A 1x3x3 averaged representation of all ROIs
    landmarks:     np.ndarray          # (468, 3) landmark array
    face_bbox:     tuple               # (x, y, w, h) bounding box of entire face
    quality_score: float               # 0–1 face detection quality
    annotated_frame: np.ndarray        # frame with drawn landmarks


class FaceROIExtractor:
    """
    Extracts skin ROIs from face for rPPG signal measurement
    """

    # Facemesh gives around 468 points on the face, we are taking only the useful ones  
    # Landmark clusters for each ROI
    FOREHEAD_LANDMARKS = [
        10, 67, 69, 104, 108, 151, 337, 338, 297, 299, 333,
        9, 8, 107, 55, 285, 109, 103, 332, 293, 168
    ]
 
    # CHANGE: Expanded cheek clusters to include more stable mid-cheek points.
    # Removed jaw-edge points (213, 433) — they have high motion artifact.
    LEFT_CHEEK_LANDMARKS  = [
        116, 117, 118, 119, 100, 126, 142, 203, 206, 207,
        187, 147, 123, 50, 36, 192, 214
    ]
    RIGHT_CHEEK_LANDMARKS = [
        345, 346, 347, 348, 329, 355, 371, 423, 426, 427,
        411, 376, 352, 280, 266, 416, 434
    ]
 
    # NEW: Nose bridge ROI — very stable, low motion, good blood perfusion
    NOSE_LANDMARKS = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13]

    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, frame_bgr: np.ndarray) -> Optional[FaceROI]:
        """
        Process a BGR frame and return FaceROI or None if no face detected.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_lm = results.multi_face_landmarks[0]
        landmarks = np.array(
            [(lm.x * w, lm.y * h, lm.z) for lm in face_lm.landmark],
            dtype=np.float32,
        )

        # Smooth landmarks to reduce jitter
        if hasattr(self, "prev_landmarks"):
            landmarks = 0.8 * self.prev_landmarks + 0.2 * landmarks

        self.prev_landmarks = landmarks

        # Extract exact mean colors using polygon masks instead of rectangular crops
        forehead   = self._extract_roi_mean(frame_rgb, landmarks, self.FOREHEAD_LANDMARKS)
        left_cheek = self._extract_roi_mean(frame_rgb, landmarks, self.LEFT_CHEEK_LANDMARKS)
        right_cheek= self._extract_roi_mean(frame_rgb, landmarks, self.RIGHT_CHEEK_LANDMARKS)
        nose        = self._extract_roi_mean(frame_rgb, landmarks, self.NOSE_LANDMARKS)

        if forehead is None or left_cheek is None or right_cheek is None:
            return None

        # Combined: weighted mean (forehead has strongest rPPG signal)
        combined = self._combine_rois(forehead, left_cheek, right_cheek)

        # Bounding box
        xs = landmarks[:, 0]; ys = landmarks[:, 1]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        bbox = (x1, y1, x2 - x1, y2 - y1)

        # Quality: face fill ratio
        face_area = (x2 - x1) * (y2 - y1)
        quality   = float(np.clip(face_area / (w * h) * 5, 0, 1))

        # Annotated frame
        annotated = self._draw_annotations(frame_bgr.copy(), landmarks,
                                           face_lm, (h, w))

        return FaceROI(
            forehead_rgb=forehead,
            left_cheek_rgb=left_cheek,
            right_cheek_rgb=right_cheek,
            combined_rgb=combined,
            landmarks=landmarks,
            face_bbox=bbox,
            quality_score=quality,
            annotated_frame=annotated,
        )

    def release(self):
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except Exception:
            pass   # Already closed or similar issue

    @staticmethod
    def _extract_roi_mean(frame_rgb: np.ndarray, landmarks: np.ndarray,
                     indices: list) -> Optional[np.ndarray]:
        """Extract exact mean color using a precisely masked convex hull."""
        h, w = frame_rgb.shape[:2]
        pts  = landmarks[indices, :2].astype(np.int32)
        
        # Create a blank mask and draw a precise polygon around the landmark cluster
        mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

        # Convert to YCrCb for better skin detection
        ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        # Skin color mask
        skin_mask = ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135)).astype(np.uint8) * 255

        # Combine ROI mask + skin mask
        final_mask = cv2.bitwise_and(mask, skin_mask)

        # Use final mask
        if cv2.countNonZero(final_mask) > 0:
            mean_color = cv2.mean(frame_rgb, mask=final_mask)[:3]
            return np.array(mean_color, dtype=np.float64).reshape(1, 1, 3)
        
        # Calculate mathematically accurate mean of ONLY the skin pixels inside the mask
        if cv2.countNonZero(mask) > 0:
            mean_color = cv2.mean(frame_rgb, mask=mask)[:3]
            return np.array(mean_color, dtype=np.float64).reshape(1, 1, 3)
        return None

    @staticmethod
    def _combine_rois(forehead: np.ndarray, left_cheek: np.ndarray, right_cheek: np.ndarray) -> np.ndarray:
        """Combine ROIs dynamically with optimized physiological weighting."""
        m_f = forehead.flatten()
        m_l = left_cheek.flatten()
        m_r = right_cheek.flatten()
        
        # The forehead typically contains a stronger and less distorted blood volume pulse signal 
        # compared to the cheeks (which are more susceptible to movement and geometry distortions).
        combined_mean = (m_f * 0.5) + (m_l * 0.25) + (m_r * 0.25)
        
        return combined_mean.reshape(1, 1, 3)

    def _draw_annotations(self, frame: np.ndarray, landmarks: np.ndarray,
                          face_lm, shape) -> np.ndarray:
        """Return the clean frame without drawing the mesh or ROI boxes."""
        return frame