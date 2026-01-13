"""
姿勢推定モジュール
MediaPipe Poseを使用して動画から姿勢を検出します
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional


class PoseEstimator:
    """MediaPipe Poseを使用した姿勢推定クラス"""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        初期化

        Args:
            min_detection_confidence: 検出の最小信頼度
            min_tracking_confidence: トラッキングの最小信頼度
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """
        1フレームを処理して姿勢を検出

        Args:
            frame: 入力画像（BGR形式）

        Returns:
            処理済み画像とpose_landmarksのタプル
        """
        # BGRをRGBに変換
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # 姿勢推定を実行
        results = self.pose.process(image_rgb)

        # 描画のために書き込み可能に
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # ランドマークを描画
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return image_bgr, results.pose_landmarks

    def extract_keypoints(self, pose_landmarks, image_shape: Tuple[int, int]) -> Dict[str, Tuple[float, float]]:
        """
        ランドマークから主要なキーポイントを抽出

        Args:
            pose_landmarks: MediaPipeのpose_landmarks
            image_shape: 画像の形状 (height, width)

        Returns:
            キーポイント名と座標の辞書
        """
        if not pose_landmarks:
            return {}

        height, width = image_shape
        keypoints = {}

        # 主要なランドマークのインデックス
        landmark_indices = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        }

        for name, landmark_idx in landmark_indices.items():
            landmark = pose_landmarks.landmark[landmark_idx]
            # 正規化座標をピクセル座標に変換
            x = landmark.x * width
            y = landmark.y * height
            keypoints[name] = (x, y)

        return keypoints

    def calculate_angle(self, point1: Tuple[float, float],
                       point2: Tuple[float, float],
                       point3: Tuple[float, float]) -> float:
        """
        3点から角度を計算

        Args:
            point1: 最初の点 (x, y)
            point2: 中心の点 (x, y)
            point3: 最後の点 (x, y)

        Returns:
            角度（度数法）
        """
        # ベクトルを計算
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)

        # 角度を計算
        cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))

        return np.degrees(angle)

    def close(self):
        """リソースを解放"""
        self.pose.close()
