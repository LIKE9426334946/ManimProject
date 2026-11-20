"""在仓库根目录运行 `uv run manim -pql 01_Linear/vector_distance.py VectorDistance`。"""

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UL, UP, UR, Arrow, Create, DecimalNumber, FadeIn,
    FadeOut, GrowArrow, Line, NumberPlane, Scene, Text, ValueTracker, VGroup,
    always_redraw, interpolate, smooth,
)


class VectorDistance(Scene):
    def construct(self):
        # 两个向量共用同一个矩阵。A 只压缩水平方向，保留纵坐标。
        u = np.array([-5.0, 3.0])
        v = np.array([5.0, 3.0])
        matrix = np.array([[0.1, 0.0], [0.0, 1.0]])

        self.camera.background_color = "#101722"
        u_color, v_color, distance_color = "#64D8D0", "#FF977D", "#FFD076"

        # 横纵轴每一格等长，整个动画使用同一个固定坐标系。
        plane = NumberPlane(
            x_range=[-6, 6, 1], y_range=[-1, 4, 1],
            x_length=11.4, y_length=4.75, faded_line_ratio=0,
            axis_config={"color": "#A4B2C5", "font_size": 20, "label_constructor": Text},
            x_axis_config={"numbers_to_include": [-4, -2, 2, 4], "label_direction": DOWN},
            y_axis_config={"numbers_to_include": [2, 4], "label_direction": LEFT},
            background_line_style={"stroke_color": "#425269", "stroke_width": 1.3},
        ).move_to(0.85 * DOWN)
        x_label = Text("x", font_size=25).next_to(plane.c2p(6, 0), RIGHT, buff=0.12)
        y_label = Text("y", font_size=25).next_to(plane.c2p(0, 4), UP, buff=0.12)

        # 矩阵元素直接取自参与计算的数组。所有文字都用 Pango 绘制。
        entries = VGroup(*(Text(f"{n:g}", font_size=29) for n in matrix.flat))
        entries.arrange_in_grid(rows=2, cols=2, buff=(0.32, 0.14))
        brackets = [Text(mark).stretch_to_fit_height(entries.height + 0.15) for mark in "[]"]
        matrix_view = VGroup(
            Text("变换矩阵   A =", font_size=29), brackets[0], entries, brackets[1],
        ).arrange(RIGHT, buff=0.12)

        distance = DecimalNumber(
            np.linalg.norm(v - u), num_decimal_places=2, mob_class=Text,
            font_size=38, color=distance_color,
        )
        distance_view = VGroup(
            Text("端点距离", font_size=29), distance,
        ).arrange(RIGHT, buff=0.25)
        header = VGroup(matrix_view, distance_view).arrange(RIGHT, buff=2.1)
        header.to_edge(UP, buff=0.48)

        # t 从 0 走到 1，每一帧都对两个向量使用同一变换 (1-t)I + tA。
        progress = ValueTracker(0)

        def position(vector):
            return interpolate(vector, matrix @ vector, progress.get_value())

        def tip(vector):
            return plane.c2p(*position(vector))

        origin = plane.c2p(0, 0)

        def moving_arrow(vector, color):
            return always_redraw(lambda: Arrow(
                origin, tip(vector), buff=0, color=color, stroke_width=6, tip_length=0.23,
            ))

        arrow_u = moving_arrow(u, u_color)
        arrow_v = moving_arrow(v, v_color)

        # 金线连接两个箭头尖端，读数使用它们在坐标系里的欧氏距离。
        span = always_redraw(lambda: Line(
            tip(u), tip(v), color=distance_color, stroke_width=3,
        ))
        distance.add_updater(lambda number: number.set_value(
            np.linalg.norm(position(v) - position(u))
        ))

        def coordinate_label(name, point, direction, color):
            x, y = point
            label = Text(f"{name} = ({x:g}, {y:g})", font_size=26, color=color)
            return label.next_to(plane.c2p(x, y), direction, buff=0.14)

        before = VGroup(
            coordinate_label("u", u, UL, u_color),
            coordinate_label("v", v, UR, v_color),
        )
        after = VGroup(
            coordinate_label("Au", matrix @ u, UL, u_color),
            coordinate_label("Av", matrix @ v, UR, v_color),
        )

        self.play(FadeIn(plane), FadeIn(x_label), FadeIn(y_label), FadeIn(header))
        self.play(GrowArrow(arrow_u), GrowArrow(arrow_v))
        self.play(Create(span), FadeIn(before))
        self.wait(1)
        self.play(FadeOut(before), run_time=0.35)
        self.play(progress.animate.set_value(1), run_time=4, rate_func=smooth)
        self.play(FadeIn(after))
        self.wait(2)
