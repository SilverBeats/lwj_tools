#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


class PlotType(Enum):
    LINE = 'Line'
    BAR = 'Bar'
    SCATTER = 'Scatter'
    HEATMAP = 'Heatmap'


@dataclass
class FontConfig:
    """
    字体配置
    Args:
        family: 字体族
        size: 字号(pt)
        weight: 字重，比如 bold
        style: 字体风格，比如 normal, italic
        color: css 颜色或者十六进制
    """
    family: str = 'SimHei'
    size: int = 12
    weight: str = 'normal'  # normal, bold, etc.
    style: str = 'normal'
    color: str = 'black'


@dataclass
class LegendConfig:
    """
    图例配置
    Args:
        enabled: 是否显示图例
        loc: 图例的位置
        font_config: 图例的字体配置
        frame_on: 图例边框
        shadow: 是否启用阴影
        border_pad: 图例边框内边距
        column_spacing: 多列图例时列间距
        handle_text_pad: 图例符号与文字间距
        n_col: 图例的列数
    """
    enabled: bool = True
    loc: str = 'best'
    font_config: FontConfig = field(default_factory=FontConfig)
    frame_on: bool = True
    shadow: bool = False
    border_pad: float = 0.4
    column_spacing: float = 2.0
    handle_text_pad: float = 0.8
    n_col: int = 1


@dataclass
class AxConfig:
    """
    坐标轴配置
    Args:
        ticks: 自定义刻度位置，比如 [0, 2, 4, 6]
        tick_labels: 刻度对应的文本标签，比如 ['一月', '二月', ...]
        tick_font: 刻度文字的字体配置
        tick_rotation: 刻度 label 的旋转度数
        grid_enabled: 是否启动网格线
        grid_alpha: 网格线透明度
        lim: 坐标轴范围
        label: 坐标轴标题，比如 销售额
        label_font: 坐标轴标题的字体配置
    """
    ticks: Optional[List[float]] = None
    tick_labels: Optional[List[str]] = None
    tick_font: FontConfig = field(default_factory=FontConfig)
    tick_rotation: int = 0
    grid_enabled: bool = True
    grid_alpha: float = 0.3
    lim: Optional[Tuple[float, float]] = None
    label: str = ""
    label_font: FontConfig = field(default_factory=FontConfig)


@dataclass
class DataPointConfig:
    """
    数据点配置
    Args:
        show_label: 是否在图上显示具体数值
        font: 数据标签的字体配置
        fmt_fn: 数值标签格式化方法
        offset_x: 数据标签相对于数据点的 x 轴偏移量
        offset_y: 数据标签相对于数据点的 y 轴偏移量
    """
    show_label: bool = False
    font: FontConfig = field(default_factory=FontConfig)
    color: str = "blue"
    fmt_fn: Callable[[float], str] = None
    offset_x: float = 0.0
    offset_y: float = 0.0

    def __post_init__(self):
        if self.fmt_fn is None:
            self.fmt_fn = lambda x: f'{x:.2f}'


@dataclass
class TitleConfig:
    """
    标题配置
    Args:
        text: 标题内容
        font: 标题字体
        pad: 标题与图表顶部的间距 (pt)
        loc: 对齐方式 left, center, right
    """
    text: str = ""
    font: FontConfig = field(default_factory=FontConfig)
    pad: float = 20.0
    loc: str = 'center'


@dataclass
class ColorBarConfig:
    """
    用于热力图
    Args:
        enabled: 是否显示颜色条
        label: 颜色条标题
        loc: 颜色条位置
        shrink: 颜色条缩放比例（<1 表示缩短）
        aspect: 颜色条宽高比（越大越细长）
    """
    enabled: bool = True
    label: str = ""
    loc: str = "right"
    shrink: float = 0.9
    aspect: float = 20


@dataclass
class SeriesConfig:
    """
    单个数据系列配置
    Args:
        x_data: 横轴数据
        y_data: 纵轴数据
        plot_type: 绘图类型
        label: 用于图例
        color: 柱子/折线的颜色
        alpha: 透明度 (0-1)
        width: 柱子的宽度/散点的大小/折线的粗细
        marker: 散点图/折线图标记的形状
        data_point_config: 数据标签配置
    """
    x_data: Union[List, np.ndarray]
    y_data: Union[List, np.ndarray]
    plot_type: PlotType
    label: str = ""
    color: str = "auto"
    alpha: float = 1.0
    width: float = 1.0
    marker: str = 'o'
    data_point_config: DataPointConfig = field(default_factory=DataPointConfig)
    color_bar_config: Optional[ColorBarConfig] = None


@dataclass
class SubplotConfig:
    """子图配置"""
    series_list: List[SeriesConfig]
    title_config: TitleConfig = field(default_factory=TitleConfig)
    x_axis_config: AxConfig = field(default_factory=AxConfig)
    y_axis_config: AxConfig = field(default_factory=AxConfig)
    legend_config: LegendConfig = field(default_factory=LegendConfig)


@dataclass
class PlotConfig:
    """
    综合绘图配置
    Args:
        subplots: 子图的配置
        layout: (rows, cols)
        figure_size:
        dpi:
        save_path: 图片保存路径
        show_plot: 是否调用 plt.show()
        main_title: 主标题
        main_title_config: 主标题的配置
    """
    subplots: List[SubplotConfig]
    layout: Tuple[int, int] = (1, 1)
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 300
    save_path: Optional[str] = None
    show_plot: bool = True
    main_title: Optional[str] = None
    main_title_config: TitleConfig = field(default_factory=TitleConfig)

    def __post_init__(self):
        max_capacity = self.layout[0] * self.layout[1]
        if len(self.subplots) > max_capacity:
            raise ValueError(f'子图数量（{len(self.subplots)}）超出了布局容量（{max_capacity}）')


class EasyPlot:
    def __init__(self):
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = [
            'SimHei', 'Microsoft YaHei', 'PingFang SC',
            'Heiti TC', 'WenQuanYi Micro Hei'
        ]

    def _apply_font_config(self, font_config: FontConfig):
        """应用字体配置"""
        font_properties = FontProperties(
            family=font_config.family,
            size=font_config.size,
            weight=font_config.weight,
            style=font_config.style,
        )
        return font_properties

    def _get_default_colors(self, n: int) -> List[str]:
        """获取默认颜色列表"""
        if n <= 10:
            return plt.cm.tab10(np.linspace(0, 1, n)).tolist()
        else:
            return plt.cm.Set3(np.linspace(0, 1, n)).tolist()

    def _draw_single_bar(
        self, series: SeriesConfig, ax: plt.Axes
    ):
        bars = ax.bar(
            x=series.x_data,
            height=series.y_data,
            label=series.label,
            color=series.color,
            width=series.width,
            alpha=series.alpha,
        )

        if not series.data_point_config.show_label:
            return

        offset_x = series.data_point_config.offset_x
        offset_y = series.data_point_config.offset_y
        fmt_fn = series.data_point_config.fmt_fn
        font = series.data_point_config.font

        for bar in bars:
            height = bar.get_height()
            ax.text(
                x=bar.get_x() + bar.get_width() / 2 + offset_x,
                y=height + offset_y,
                s=fmt_fn(height),
                ha='center',
                va='bottom',
                fontproperties=self._apply_font_config(font),
                color=font.color,
            )

    def _draw_single_line(
        self, series: SeriesConfig, ax: plt.Axes
    ):
        line = ax.plot(
            series.x_data, series.y_data,
            label=series.label,
            color=series.color,
            linewidth=series.width,
            alpha=series.alpha,
            marker=series.marker,
        )[0]

        if not series.data_point_config.show_label:
            return

        offset_x = series.data_point_config.offset_x
        offset_y = series.data_point_config.offset_y
        fmt_fn = series.data_point_config.fmt_fn
        font = series.data_point_config.font

        for xi, yi in zip(series.x_data, series.y_data):
            ax.text(
                x=xi + offset_x,
                y=yi + offset_y,
                s=fmt_fn(yi),
                ha='center',
                va='bottom',
                fontproperties=self._apply_font_config(font),
                color=font.color,
            )

    def _draw_single_scatter(
        self, series: SeriesConfig, ax: plt.Axes
    ):
        scatter = ax.scatter(
            series.x_data,
            series.y_data,
            label=series.label,
            s=series.width,
            alpha=series.alpha,
            marker=series.marker,
        )

        if not series.data_point_config.show_label:
            return

        offset_x = series.data_point_config.offset_x
        offset_y = series.data_point_config.offset_y
        fmt_fn = series.data_point_config.fmt_fn
        font = series.data_point_config.font

        for xi, yi in zip(series.x_data, series.y_data):
            ax.text(
                x=xi + offset_x,
                y=yi + offset_y,
                s=fmt_fn(yi),
                ha='center',
                va='bottom',
                fontproperties=self._apply_font_config(font),
                color=font.color,
            )

    def _draw_single_heatmap(
        self, series: SeriesConfig, ax: plt.Axes
    ):
        im = ax.imshow(series.y_data, cmap='viridis', aspect='auto')

        if series.color_bar_config and series.color_bar_config.enabled:
            color_bar_config = series.color_bar_config
            cbar = ax.figure.colorbar(
                im,
                ax=ax,
                shrink=color_bar_config.shrink,
                aspect=color_bar_config.aspect,
            )
            if color_bar_config.label:
                cbar.set_label(color_bar_config.label)

    def _draw_single_series(
        self,
        series: SeriesConfig,
        ax: plt.Axes,
    ):
        if series.plot_type == PlotType.BAR:
            self._draw_single_bar(series, ax)
        elif series.plot_type == PlotType.LINE:
            self._draw_single_line(series, ax)
        elif series.plot_type == PlotType.SCATTER:
            self._draw_single_scatter(series, ax)
        elif series.plot_type == PlotType.HEATMAP:
            self._draw_single_heatmap(series, ax)
        else:
            raise ValueError(f'未知 PlotType: {series.plot_type}')

    def _configure_axes(self, subplot_config: SubplotConfig, ax: plt.Axes):
        # 设置子图标题
        if subplot_config.title_config.text:
            ax.set_title(
                subplot_config.title_config.text,
                fontproperties=self._apply_font_config(subplot_config.title_config.font),
                pad=subplot_config.title_config.pad,
                loc=subplot_config.title_config.loc,
                color=subplot_config.title_config.font.color,
            )

        # 设置X轴
        if subplot_config.x_axis_config.ticks is not None:
            ax.set_xticks(subplot_config.x_axis_config.ticks)
        if subplot_config.x_axis_config.tick_labels is not None:
            ax.set_xticklabels(subplot_config.x_axis_config.tick_labels)
        if subplot_config.x_axis_config.lim is not None:
            ax.set_xlim(subplot_config.x_axis_config.lim)
        if subplot_config.x_axis_config.label:
            ax.set_xlabel(
                subplot_config.x_axis_config.label,
                fontproperties=self._apply_font_config(subplot_config.x_axis_config.label_font),
            )

        # 设置Y轴
        if subplot_config.y_axis_config.ticks is not None:
            ax.set_yticks(subplot_config.y_axis_config.ticks)
        if subplot_config.y_axis_config.tick_labels is not None:
            ax.set_yticklabels(subplot_config.y_axis_config.tick_labels)
        if subplot_config.y_axis_config.lim is not None:
            ax.set_ylim(subplot_config.y_axis_config.lim)
        if subplot_config.y_axis_config.label:
            ax.set_ylabel(
                subplot_config.y_axis_config.label,
                fontproperties=self._apply_font_config(subplot_config.y_axis_config.label_font),
            )

        # 设置网格
        if subplot_config.x_axis_config.grid_enabled and subplot_config.y_axis_config.grid_enabled:
            ax.grid(alpha=subplot_config.x_axis_config.grid_alpha)
        elif subplot_config.x_axis_config.grid_enabled:
            ax.grid(axis='x', alpha=subplot_config.x_axis_config.grid_alpha)
        elif subplot_config.y_axis_config.grid_enabled:
            ax.grid(axis='y', alpha=subplot_config.y_axis_config.grid_alpha)

        # 应用坐标轴刻度字体配置
        ax.tick_params(
            axis='x', labelsize=subplot_config.x_axis_config.tick_font.size,
            labelcolor=subplot_config.x_axis_config.tick_font.color,
        )
        ax.tick_params(
            axis='y', labelsize=subplot_config.y_axis_config.tick_font.size,
            labelcolor=subplot_config.y_axis_config.tick_font.color,
        )

        # 设置图例
        if subplot_config.legend_config.enabled and any(s.label for s in subplot_config.series_list):
            ax.legend(
                loc=subplot_config.legend_config.loc,
                prop=self._apply_font_config(subplot_config.legend_config.font_config),
                frameon=subplot_config.legend_config.frame_on,
                shadow=subplot_config.legend_config.shadow,
                borderaxespad=subplot_config.legend_config.border_pad,
                columnspacing=subplot_config.legend_config.column_spacing,
                handletextpad=subplot_config.legend_config.handle_text_pad,
                ncol=subplot_config.legend_config.n_col,
            )

    def draw(self, config: PlotConfig):
        rows, cols = config.layout
        fig, axes = plt.subplots(
            nrows=rows,
            ncols=cols,
            figsize=config.figure_size,
            dpi=config.dpi,
        )

        if rows == cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # 设置主标题
        if config.main_title:
            fig.suptitle(
                t=config.main_title,
                fontproperties=self._apply_font_config(config.main_title_config.font),
                color=config.main_title_config.font.color,
            )

        # 获取所有需要自动分配颜色的系列
        auto_series_count = 0
        for subplot_config in config.subplots:
            for series in subplot_config.series_list:
                if series.color == 'auto':
                    auto_series_count += 1

        # 分配默认颜色
        default_colors = self._get_default_colors(auto_series_count)
        auto_color_idx = 0

        # 绘制每个子图
        for i, (subplot_config, ax) in enumerate(zip(config.subplots, axes)):

            # 为 auto 部分的 series 分配颜色
            for series in subplot_config.series_list:
                if series.color == 'auto':
                    series.color = default_colors[auto_color_idx]
                    auto_color_idx += 1

            # 绘制该子图的所有系列
            for series in subplot_config.series_list:
                self._draw_single_series(series, ax)

            # 配置坐标轴
            self._configure_axes(subplot_config, ax)

        # 隐藏未使用的子图
        for i in range(len(config.subplots), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        if config.save_path:
            plt.savefig(config.save_path, dpi=config.dpi, bbox_inches='tight')

        if config.show_plot:
            plt.show()

        return fig, axes
