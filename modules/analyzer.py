"""
分析ロジックモジュール
スパイクフォームを分析し、スコアとフィードバックを生成します
"""
import numpy as np
from typing import Dict, List, Tuple


class SpikeAnalyzer:
    """スパイクフォーム分析クラス"""

    def __init__(self):
        """初期化"""
        # 理想的な角度の基準値
        self.ideal_knee_angle_takeoff = 90  # 踏み込み時の膝の角度（度）
        self.ideal_elbow_angle_hitting = 140  # 打点での肘の角度（度）
        self.ideal_arm_angle_preparation = 160  # 準備動作での腕の角度（度）

    def analyze_spike_form(self, keypoints_sequence: List[Dict], key_frames: Dict) -> Dict:
        """
        スパイクフォームを総合的に分析

        Args:
            keypoints_sequence: キーポイントのシーケンス
            key_frames: 重要フレームの辞書

        Returns:
            分析結果の辞書
        """
        analysis_results = {
            'overall_score': 0,
            'phase_scores': {},
            'feedback': [],
            'measurements': {}
        }

        if not keypoints_sequence:
            return analysis_results

        # 各フェーズを分析
        takeoff_score, takeoff_feedback = self._analyze_takeoff(key_frames)
        hitting_score, hitting_feedback = self._analyze_hitting_point(key_frames)
        jump_score, jump_feedback = self._analyze_jump_height(keypoints_sequence)

        # フェーズごとのスコアを記録
        analysis_results['phase_scores'] = {
            '踏み込み': takeoff_score,
            '打点': hitting_score,
            'ジャンプ': jump_score
        }

        # 総合スコアを計算（各フェーズの平均）
        analysis_results['overall_score'] = np.mean([takeoff_score, hitting_score, jump_score])

        # フィードバックをまとめる
        analysis_results['feedback'].extend(takeoff_feedback)
        analysis_results['feedback'].extend(hitting_feedback)
        analysis_results['feedback'].extend(jump_feedback)

        return analysis_results

    def _analyze_takeoff(self, key_frames: Dict) -> Tuple[float, List[str]]:
        """
        踏み込みフェーズを分析

        Args:
            key_frames: 重要フレームの辞書

        Returns:
            スコアとフィードバックのタプル
        """
        score = 50  # デフォルトスコア
        feedback = []

        if 'takeoff' not in key_frames:
            feedback.append("⚠️ 踏み込みフレームを検出できませんでした")
            return score, feedback

        keypoints = key_frames['takeoff']['keypoints']

        # 膝の角度をチェック
        if all(k in keypoints for k in ['right_hip', 'right_knee', 'right_ankle']):
            knee_angle = self._calculate_angle(
                keypoints['right_hip'],
                keypoints['right_knee'],
                keypoints['right_ankle']
            )

            angle_diff = abs(knee_angle - self.ideal_knee_angle_takeoff)

            if angle_diff < 10:
                score = 90
                feedback.append(f"✅ 踏み込み時の膝の角度が理想的です（{knee_angle:.1f}度）")
            elif angle_diff < 20:
                score = 70
                feedback.append(f"⚠️ 踏み込み時の膝の角度が少しずれています（{knee_angle:.1f}度、理想は{self.ideal_knee_angle_takeoff}度）")
            else:
                score = 50
                if knee_angle > self.ideal_knee_angle_takeoff:
                    feedback.append(f"❌ 膝の屈曲が浅すぎます（{knee_angle:.1f}度）。もっと深く膝を曲げましょう")
                else:
                    feedback.append(f"❌ 膝が曲がりすぎています（{knee_angle:.1f}度）。適度な屈曲を意識しましょう")

        return score, feedback

    def _analyze_hitting_point(self, key_frames: Dict) -> Tuple[float, List[str]]:
        """
        打点フェーズを分析

        Args:
            key_frames: 重要フレームの辞書

        Returns:
            スコアとフィードバックのタプル
        """
        score = 50  # デフォルトスコア
        feedback = []

        if 'highest_point' not in key_frames:
            feedback.append("⚠️ 最高到達点（打点）を検出できませんでした")
            return score, feedback

        keypoints = key_frames['highest_point']['keypoints']

        # 打点での腕の伸び具合をチェック
        if all(k in keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
            elbow_angle = self._calculate_angle(
                keypoints['right_shoulder'],
                keypoints['right_elbow'],
                keypoints['right_wrist']
            )

            angle_diff = abs(elbow_angle - self.ideal_elbow_angle_hitting)

            if angle_diff < 15:
                score = 90
                feedback.append(f"✅ 打点での腕の伸びが理想的です（{elbow_angle:.1f}度）")
            elif angle_diff < 30:
                score = 70
                feedback.append(f"⚠️ 打点での腕の角度が少しずれています（{elbow_angle:.1f}度）")
            else:
                score = 50
                if elbow_angle < self.ideal_elbow_angle_hitting:
                    feedback.append(f"❌ 打点で腕が曲がりすぎています（{elbow_angle:.1f}度）。もっと腕を伸ばしましょう")
                else:
                    feedback.append(f"❌ 打点で腕が伸びすぎています（{elbow_angle:.1f}度）。適度な曲げを保ちましょう")

        # 手首の位置が頭より高いかチェック
        if 'right_wrist' in keypoints and 'nose' in keypoints:
            wrist_y = keypoints['right_wrist'][1]
            nose_y = keypoints['nose'][1]

            if wrist_y < nose_y:  # y座標は上が小さい
                feedback.append("✅ 打点が高い位置にあります")
            else:
                feedback.append("⚠️ 打点をもっと高くすることを意識しましょう")
                score = max(score - 10, 30)

        return score, feedback

    def _analyze_jump_height(self, keypoints_sequence: List[Dict]) -> Tuple[float, List[str]]:
        """
        ジャンプの高さを分析

        Args:
            keypoints_sequence: キーポイントのシーケンス

        Returns:
            スコアとフィードバックのタプル
        """
        score = 50  # デフォルトスコア
        feedback = []

        # 鼻（頭部）の最低位置と最高位置を取得
        nose_positions = []
        for keypoints in keypoints_sequence:
            if 'nose' in keypoints:
                nose_positions.append(keypoints['nose'][1])

        if not nose_positions:
            feedback.append("⚠️ ジャンプの高さを測定できませんでした")
            return score, feedback

        min_y = min(nose_positions)
        max_y = max(nose_positions)
        jump_height_pixels = max_y - min_y

        # ジャンプの高さを評価（ピクセル単位、相対的な評価）
        # 画面の高さの10%以上のジャンプを高評価とする
        height_ratio = jump_height_pixels / max_y if max_y > 0 else 0

        if height_ratio > 0.15:
            score = 90
            feedback.append("✅ 素晴らしいジャンプ力です！")
        elif height_ratio > 0.10:
            score = 70
            feedback.append("⚠️ ジャンプの高さは良好ですが、さらに高く跳べる余地があります")
        else:
            score = 50
            feedback.append("❌ ジャンプの高さが不足しています。踏み込みと腕の振りを意識しましょう")

        return score, feedback

    def _calculate_angle(self, point1: Tuple[float, float],
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
        vector1 = np.array(point1) - np.array(point2)
        vector2 = np.array(point3) - np.array(point2)

        cosine = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))

        return np.degrees(angle)

    def generate_summary(self, analysis_results: Dict) -> str:
        """
        分析結果のサマリーを生成

        Args:
            analysis_results: 分析結果の辞書

        Returns:
            サマリーテキスト
        """
        overall_score = analysis_results['overall_score']

        if overall_score >= 80:
            grade = "優秀"
            comment = "素晴らしいフォームです！"
        elif overall_score >= 70:
            grade = "良好"
            comment = "良いフォームですが、さらに改善の余地があります。"
        elif overall_score >= 60:
            grade = "普通"
            comment = "基本はできていますが、いくつか改善点があります。"
        else:
            grade = "要改善"
            comment = "フォームを見直して、基本から練習しましょう。"

        summary = f"総合評価: {grade} (スコア: {overall_score:.1f}/100)\n{comment}"
        return summary
