"""
動画処理モジュール
動画の読み込み、フレーム抽出、姿勢推定の統合処理を行います
"""
import cv2
import numpy as np
import tempfile
from typing import List, Dict, Tuple, Optional
from .pose_estimation import PoseEstimator


class VideoProcessor:
    """動画処理クラス"""

    def __init__(self):
        """初期化"""
        self.pose_estimator = PoseEstimator()

    def process_video(self, video_path: str, progress_callback=None) -> Dict:
        """
        動画を処理して姿勢推定を実行

        Args:
            video_path: 動画ファイルのパス
            progress_callback: 進捗報告用のコールバック関数

        Returns:
            処理結果の辞書
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")

        # 動画情報を取得
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 結果を格納するリスト
        frames_with_pose = []
        all_keypoints = []
        frame_indices = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 姿勢推定を実行
            processed_frame, pose_landmarks = self.pose_estimator.process_frame(frame)

            # キーポイントを抽出
            if pose_landmarks:
                keypoints = self.pose_estimator.extract_keypoints(pose_landmarks, (height, width))
                all_keypoints.append(keypoints)
                frames_with_pose.append(processed_frame)
                frame_indices.append(frame_idx)

            # 進捗を報告
            if progress_callback:
                progress = (frame_idx + 1) / frame_count
                progress_callback(progress)

            frame_idx += 1

        cap.release()

        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'frames_with_pose': frames_with_pose,
            'keypoints_sequence': all_keypoints,
            'frame_indices': frame_indices
        }

    def save_processed_video(self, frames: List[np.ndarray], output_path: str, fps: int = 30):
        """
        処理済みフレームを動画として保存

        Args:
            frames: フレームのリスト
            output_path: 出力パス
            fps: フレームレート
        """
        if not frames:
            raise ValueError("保存するフレームがありません")

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            out.write(frame)

        out.release()

    def extract_key_frames(self, keypoints_sequence: List[Dict], frame_indices: List[int],
                          frames: List[np.ndarray]) -> Dict[str, Dict]:
        """
        重要なフレーム（最高到達点など）を抽出

        Args:
            keypoints_sequence: キーポイントのシーケンス
            frame_indices: フレームインデックスのリスト
            frames: フレーム画像のリスト

        Returns:
            重要フレームの辞書
        """
        if not keypoints_sequence:
            return {}

        key_frames = {}

        # 鼻（頭部）の高さを追跡
        nose_heights = []
        for keypoints in keypoints_sequence:
            if 'nose' in keypoints:
                nose_heights.append(keypoints['nose'][1])  # y座標（低いほど高い位置）
            else:
                nose_heights.append(float('inf'))

        # 最高到達点（鼻のy座標が最小の点）
        if nose_heights:
            min_y_idx = np.argmin(nose_heights)
            key_frames['highest_point'] = {
                'frame_index': frame_indices[min_y_idx],
                'frame': frames[min_y_idx],
                'keypoints': keypoints_sequence[min_y_idx],
                'description': '最高到達点'
            }

        # 踏み込みフレーム（膝の角度が最も曲がっている点）
        knee_angles = []
        for keypoints in keypoints_sequence:
            if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
                angle = self.pose_estimator.calculate_angle(
                    keypoints['right_hip'],
                    keypoints['right_knee'],
                    keypoints['right_ankle']
                )
                knee_angles.append(angle)
            else:
                knee_angles.append(180)  # 直立状態

        if knee_angles:
            min_angle_idx = np.argmin(knee_angles)
            key_frames['takeoff'] = {
                'frame_index': frame_indices[min_angle_idx],
                'frame': frames[min_angle_idx],
                'keypoints': keypoints_sequence[min_angle_idx],
                'description': '踏み込み'
            }

        return key_frames

    def close(self):
        """リソースを解放"""
        self.pose_estimator.close()
