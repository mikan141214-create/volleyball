"""
可視化モジュール
分析結果をグラフや図で可視化します
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import numpy as np


class Visualizer:
    """可視化クラス"""

    def __init__(self):
        """初期化"""
        self.colors = {
            'primary': '#FF6B35',  # オレンジ
            'secondary': '#004E89',  # ダークブルー
            'accent': '#4CAF50',  # ライトグリーン
            'warning': '#FFA726',  # 警告色
            'error': '#EF5350'  # エラー色
        }

    def create_score_gauge(self, score: float, title: str = "総合スコア") -> go.Figure:
        """
        スコアのゲージチャートを作成

        Args:
            score: スコア（0-100）
            title: チャートのタイトル

        Returns:
            Plotlyの図オブジェクト
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 24}},
            delta={'reference': 70, 'increasing': {'color': self.colors['accent']}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                'bar': {'color': self._get_score_color(score)},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 60], 'color': '#FFE0E0'},
                    {'range': [60, 70], 'color': '#FFF4E0'},
                    {'range': [70, 80], 'color': '#E0F0FF'},
                    {'range': [80, 100], 'color': '#E0FFE0'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        return fig

    def create_phase_scores_bar(self, phase_scores: Dict[str, float]) -> go.Figure:
        """
        フェーズ別スコアの棒グラフを作成

        Args:
            phase_scores: フェーズ名とスコアの辞書

        Returns:
            Plotlyの図オブジェクト
        """
        phases = list(phase_scores.keys())
        scores = list(phase_scores.values())

        colors = [self._get_score_color(score) for score in scores]

        fig = go.Figure(data=[
            go.Bar(
                x=phases,
                y=scores,
                marker_color=colors,
                text=[f'{score:.1f}' for score in scores],
                textposition='outside',
            )
        ])

        fig.update_layout(
            title='フェーズ別スコア',
            xaxis_title='フェーズ',
            yaxis_title='スコア',
            yaxis_range=[0, 100],
            height=400,
            showlegend=False
        )

        return fig

    def create_radar_chart(self, phase_scores: Dict[str, float]) -> go.Figure:
        """
        フェーズ別スコアのレーダーチャートを作成

        Args:
            phase_scores: フェーズ名とスコアの辞書

        Returns:
            Plotlyの図オブジェクト
        """
        categories = list(phase_scores.keys())
        values = list(phase_scores.values())

        # 閉じた図形にするため、最初の値を最後に追加
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            fillcolor=self.colors['primary'],
            opacity=0.6,
            line=dict(color=self.colors['primary'], width=2),
            name='現在のスコア'
        ))

        # 理想的なスコア（80点）を参照線として追加
        ideal_values = [80] * len(categories_closed)
        fig.add_trace(go.Scatterpolar(
            r=ideal_values,
            theta=categories_closed,
            fill='toself',
            fillcolor=self.colors['accent'],
            opacity=0.2,
            line=dict(color=self.colors['accent'], width=2, dash='dash'),
            name='目標スコア (80点)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=400,
            title='フェーズ別パフォーマンス'
        )

        return fig

    def create_history_chart(self, history_data: List[Dict]) -> go.Figure:
        """
        履歴データの折れ線グラフを作成

        Args:
            history_data: 履歴データのリスト

        Returns:
            Plotlyの図オブジェクト
        """
        if not history_data:
            fig = go.Figure()
            fig.add_annotation(
                text="履歴データがありません",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig

        dates = [item['date'] for item in history_data]
        scores = [item['score'] for item in history_data]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=10, color=self.colors['primary']),
            name='総合スコア'
        ))

        # トレンドライン（移動平均）を追加
        if len(scores) >= 3:
            window_size = min(3, len(scores))
            moving_avg = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
            fig.add_trace(go.Scatter(
                x=dates[window_size-1:],
                y=moving_avg,
                mode='lines',
                line=dict(color=self.colors['secondary'], width=2, dash='dash'),
                name='トレンド'
            ))

        fig.update_layout(
            title='スコア推移',
            xaxis_title='日付',
            yaxis_title='スコア',
            yaxis_range=[0, 100],
            height=400,
            hovermode='x unified'
        )

        return fig

    def _get_score_color(self, score: float) -> str:
        """
        スコアに応じた色を取得

        Args:
            score: スコア（0-100）

        Returns:
            カラーコード
        """
        if score >= 80:
            return self.colors['accent']  # 緑
        elif score >= 70:
            return self.colors['primary']  # オレンジ
        elif score >= 60:
            return self.colors['warning']  # 黄色
        else:
            return self.colors['error']  # 赤
