"""
バレーボールスパイクフォーム分析アプリ
Streamlit メインアプリケーション
"""
import streamlit as st
import cv2
import tempfile
import os
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from PIL import Image

from modules.video_processor import VideoProcessor
from modules.analyzer import SpikeAnalyzer
from modules.visualizer import Visualizer


# ページ設定
st.set_page_config(
    page_title="Spike Master - スパイクフォーム分析",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #004E89;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #E55A2A;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """セッション状態を初期化"""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None


def save_uploaded_file(uploaded_file):
    """アップロードされたファイルを一時保存"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"ファイルの保存に失敗しました: {e}")
        return None


def analyze_video(video_path):
    """動画を分析"""
    try:
        # プログレスバーを表示
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"処理中... {int(progress * 100)}%")

        # 動画処理
        status_text.text("動画を読み込んでいます...")
        processor = VideoProcessor()

        status_text.text("姿勢推定を実行中...")
        video_data = processor.process_video(video_path, progress_callback=update_progress)

        status_text.text("重要フレームを抽出中...")
        key_frames = processor.extract_key_frames(
            video_data['keypoints_sequence'],
            video_data['frame_indices'],
            video_data['frames_with_pose']
        )

        # 分析
        status_text.text("フォームを分析中...")
        analyzer = SpikeAnalyzer()
        analysis_results = analyzer.analyze_spike_form(
            video_data['keypoints_sequence'],
            key_frames
        )

        # 完了
        progress_bar.progress(1.0)
        status_text.text("分析完了!")

        # クリーンアップ
        processor.close()

        return {
            'video_data': video_data,
            'key_frames': key_frames,
            'analysis_results': analysis_results,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        st.error(f"分析中にエラーが発生しました: {e}")
        return None


def display_analysis_results(analysis):
    """分析結果を表示"""
    if not analysis:
        return

    results = analysis['analysis_results']
    key_frames = analysis['key_frames']

    # ヘッダー
    st.markdown("---")
    st.markdown("## 📊 分析結果")

    # 総合スコア
    col1, col2 = st.columns([1, 2])

    with col1:
        visualizer = Visualizer()
        gauge_fig = visualizer.create_score_gauge(results['overall_score'])
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col2:
        st.markdown("### 総合評価")
        analyzer = SpikeAnalyzer()
        summary = analyzer.generate_summary(results)
        st.info(summary)

        st.markdown("### 改善ポイント")
        for feedback in results['feedback']:
            st.markdown(f"- {feedback}")

    # フェーズ別スコア
    st.markdown("---")
    st.markdown("## 📈 フェーズ別分析")

    col1, col2 = st.columns(2)

    with col1:
        bar_fig = visualizer.create_phase_scores_bar(results['phase_scores'])
        st.plotly_chart(bar_fig, use_container_width=True)

    with col2:
        radar_fig = visualizer.create_radar_chart(results['phase_scores'])
        st.plotly_chart(radar_fig, use_container_width=True)

    # 重要フレーム
    st.markdown("---")
    st.markdown("## 🎯 重要フレーム")

    if key_frames:
        cols = st.columns(len(key_frames))
        for idx, (frame_name, frame_data) in enumerate(key_frames.items()):
            with cols[idx]:
                st.markdown(f"**{frame_data['description']}**")
                # BGRからRGBに変換
                frame_rgb = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_container_width=True)
    else:
        st.warning("重要フレームを検出できませんでした")


def display_history():
    """履歴を表示"""
    st.markdown("---")
    st.markdown("## 📚 分析履歴")

    if not st.session_state.analysis_history:
        st.info("まだ分析履歴がありません。動画をアップロードして分析を開始しましょう！")
        return

    # 履歴データを準備
    history_data = []
    for idx, analysis in enumerate(st.session_state.analysis_history):
        timestamp = datetime.fromisoformat(analysis['timestamp'])
        history_data.append({
            'date': timestamp.strftime('%Y-%m-%d %H:%M'),
            'score': analysis['analysis_results']['overall_score']
        })

    # グラフを表示
    visualizer = Visualizer()
    history_fig = visualizer.create_history_chart(history_data)
    st.plotly_chart(history_fig, use_container_width=True)

    # 履歴リスト
    st.markdown("### 過去の分析")
    for idx, analysis in enumerate(reversed(st.session_state.analysis_history)):
        timestamp = datetime.fromisoformat(analysis['timestamp'])
        score = analysis['analysis_results']['overall_score']

        with st.expander(f"📅 {timestamp.strftime('%Y-%m-%d %H:%M')} - スコア: {score:.1f}/100"):
            display_analysis_results(analysis)


def main():
    """メイン関数"""
    initialize_session_state()

    # ヘッダー
    st.markdown('<h1 class="main-header">🏐 Spike Master</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">バレーボールスパイクフォーム分析アプリ</p>', unsafe_allow_html=True)

    # サイドバー
    with st.sidebar:
        st.markdown("## 📖 使い方")
        st.markdown("""
        1. **動画をアップロード**: スパイクの動画（MP4形式）をアップロードします
        2. **分析開始**: アップロード後、自動的に分析が開始されます
        3. **結果確認**: スコアと改善ポイントを確認します
        4. **履歴管理**: 過去の分析結果を比較できます

        ### 📹 撮影のコツ
        - **角度**: 真横から撮影
        - **距離**: 3〜5メートル
        - **時間**: 3〜10秒程度
        - **画質**: できるだけ高画質で
        """)

        st.markdown("---")
        st.markdown("### ⚙️ 設定")

        if st.button("履歴をクリア"):
            st.session_state.analysis_history = []
            st.session_state.current_analysis = None
            st.success("履歴をクリアしました")
            st.rerun()

    # メインコンテンツ
    tab1, tab2, tab3 = st.tabs(["🎥 新規分析", "📊 分析結果", "📈 履歴"])

    with tab1:
        st.markdown("## 動画をアップロード")

        uploaded_file = st.file_uploader(
            "スパイク動画を選択してください（MP4形式）",
            type=['mp4', 'mov', 'avi'],
            help="側面から撮影された3〜10秒程度の動画が最適です"
        )

        if uploaded_file is not None:
            # 動画を一時保存
            video_path = save_uploaded_file(uploaded_file)

            if video_path:
                # 動画プレビュー
                st.video(uploaded_file)

                # 分析ボタン
                if st.button("🔍 分析を開始", type="primary"):
                    with st.spinner("分析中..."):
                        analysis = analyze_video(video_path)

                        if analysis:
                            st.session_state.current_analysis = analysis
                            st.session_state.analysis_history.append(analysis)
                            st.success("分析が完了しました！「分析結果」タブで確認してください。")
                            st.balloons()

                        # 一時ファイルを削除
                        try:
                            os.unlink(video_path)
                        except:
                            pass

    with tab2:
        if st.session_state.current_analysis:
            display_analysis_results(st.session_state.current_analysis)
        else:
            st.info("まだ分析結果がありません。「新規分析」タブで動画をアップロードしてください。")

    with tab3:
        display_history()

    # フッター
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">© 2026 Spike Master | バレーボール上達支援アプリ</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
