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
import yt_dlp
import re

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
        # ファイルポインタを先頭に戻す
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"ファイルの保存に失敗しました: {e}")
        return None


def is_valid_youtube_url(url):
    """YouTube URLが有効かチェック（通常動画とショート動画の両方に対応）"""
    # YouTubeの各種URL形式に対応
    # - 通常: https://www.youtube.com/watch?v=xxxxx
    # - 短縮: https://youtu.be/xxxxx
    # - ショート: https://www.youtube.com/shorts/xxxxx
    # - 埋め込み: https://www.youtube.com/embed/xxxxx
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|shorts/|.+\?v=)?([^&=%\?]{11})'
    match = re.match(youtube_regex, url)
    return bool(match)


def download_youtube_video(url):
    """YouTube動画をダウンロード"""
    try:
        # 一時ファイルを作成
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'video.mp4')

        # yt-dlpのオプション設定
        ydl_opts = {
            'format': 'best[ext=mp4][height<=720]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        # 動画をダウンロード
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            # ダウンロードされたファイルのパスを確認
            if os.path.exists(output_path):
                return output_path

            # 拡張子が異なる場合があるので検索
            for file in os.listdir(temp_dir):
                if file.startswith('video'):
                    return os.path.join(temp_dir, file)

            raise Exception("動画のダウンロードに失敗しました")

    except Exception as e:
        st.error(f"YouTube動画のダウンロードエラー: {e}")
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

        # プログレスバーとステータステキストをクリア
        progress_bar.empty()
        status_text.empty()

        return {
            'video_data': video_data,
            'key_frames': key_frames,
            'analysis_results': analysis_results,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        st.error(f"分析中にエラーが発生しました: {e}")
        # エラー時もプログレスバーとステータステキストをクリア
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass
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
        # 最大2列に制限
        num_frames = len(key_frames)
        num_cols = min(num_frames, 2)
        cols = st.columns(num_cols)

        for idx, (frame_name, frame_data) in enumerate(key_frames.items()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.markdown(f"**{frame_data['description']}**")
                # BGRからRGBに変換
                frame_rgb = cv2.cvtColor(frame_data['frame'], cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_container_width=True, caption=frame_data['description'])
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
    history_list = list(reversed(st.session_state.analysis_history))
    for idx, analysis in enumerate(history_list):
        timestamp = datetime.fromisoformat(analysis['timestamp'])
        score = analysis['analysis_results']['overall_score']

        with st.expander(
            f"📅 {timestamp.strftime('%Y-%m-%d %H:%M')} - スコア: {score:.1f}/100",
            expanded=False
        ):
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
        1. **動画を選択**:
           - ファイルアップロード、または
           - YouTube URLを入力
        2. **分析開始**: 動画取得後、分析ボタンをクリック
        3. **結果確認**: スコアと改善ポイントを確認
        4. **履歴管理**: 過去の分析結果を比較

        ### 📹 動画のポイント
        - **角度**: 真横から撮影
        - **距離**: 3〜5メートル
        - **時間**: 3〜10秒程度
        - **画質**: できるだけ高画質で

        ### 🔗 YouTube URLについて
        - 通常動画とショート動画の両方に対応
        - 公開動画のみ対応
        - 短い動画（3〜10秒）推奨
        - 埋め込み無効の動画は不可
        """)

        st.markdown("---")
        st.markdown("### ⚙️ 設定")

        clear_button = st.button("履歴をクリア")
        if clear_button:
            st.session_state.analysis_history = []
            st.session_state.current_analysis = None
            st.success("履歴をクリアしました")

    # メインコンテンツ
    tab1, tab2, tab3 = st.tabs(["🎥 新規分析", "📊 分析結果", "📈 履歴"])

    with tab1:
        st.markdown("## 動画を選択")

        # 動画取得方法を選択
        input_method = st.radio(
            "動画の取得方法を選択してください",
            ["📁 ファイルアップロード", "🔗 YouTube URL"],
            key="input_method"
        )

        video_path = None

        if input_method == "📁 ファイルアップロード":
            st.markdown("### ファイルをアップロード")
            uploaded_file = st.file_uploader(
                "スパイク動画を選択してください（MP4形式）",
                type=['mp4', 'mov', 'avi'],
                help="側面から撮影された3〜10秒程度の動画が最適です",
                key="video_uploader"
            )

            if uploaded_file is not None:
                # 動画プレビュー
                st.video(uploaded_file)

                # 動画を一時保存
                video_path = save_uploaded_file(uploaded_file)

        else:  # YouTube URL
            st.markdown("### YouTube URLを入力")
            youtube_url = st.text_input(
                "YouTube動画のURLを入力してください（通常動画・ショート動画対応）",
                placeholder="https://www.youtube.com/watch?v=... または https://www.youtube.com/shorts/...",
                help="YouTubeの動画URLを貼り付けてください（通常動画・ショート動画の両方に対応）",
                key="youtube_url"
            )

            if youtube_url:
                if is_valid_youtube_url(youtube_url):
                    # プレビュー用にYouTube埋め込み表示
                    st.video(youtube_url)

                    # ダウンロードボタン
                    if st.button("📥 動画をダウンロード", key="download_btn"):
                        with st.spinner("YouTube動画をダウンロード中..."):
                            video_path = download_youtube_video(youtube_url)
                            if video_path:
                                st.success("ダウンロード完了！")
                else:
                    st.error("無効なYouTube URLです。正しいURLを入力してください。")

        # 分析ボタン（動画パスが取得できた場合のみ表示）
        if video_path:
            st.markdown("---")
            analyze_button = st.button("🔍 分析を開始", type="primary", key="analyze_btn")

            if analyze_button:
                with st.spinner("分析中..."):
                    analysis = analyze_video(video_path)

                    if analysis:
                        st.session_state.current_analysis = analysis
                        st.session_state.analysis_history.append(analysis)
                        st.success("分析が完了しました！「分析結果」タブで確認してください。")

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
