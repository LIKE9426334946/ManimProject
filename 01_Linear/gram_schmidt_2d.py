"""在仓库根目录运行 `uv run manim -pqm 01_Linear/gram_schmidt_2d.py GramSchmidt2D`。"""

import numpy as np
from manim import (
    DOWN, DR, LEFT, RIGHT, UL, UP, Arrow, Circle, Create, DashedLine, Dot,
    FadeIn, FadeOut, GrowArrow, Line, NumberPlane, ReplacementTransform,
    RightAngle, Scene, Text, Transform, VGroup, smooth,
)


class GramSchmidt2D(Scene):
    def construct(self):
        v1 = np.array([4.0, 2.0])
        v2 = np.array([1.0, 3.0])

        # 保留第一个方向，用投影矩阵 P 求出 v2 沿 u1 的分量。
        u1 = v1
        projector = np.outer(u1, u1) / (u1 @ u1)
        coefficient = (v2 @ u1) / (u1 @ u1)
        projection = projector @ v2
        u2 = v2 - projection
        q1 = u1 / np.linalg.norm(u1)
        q2 = u2 / np.linalg.norm(u2)

        self.camera.background_color = "#101722"
        teal, coral, gold, violet = "#64D8D0", "#FF977D", "#FFD076", "#B6A1FF"
        white, muted = "#EEF3FA", "#A4B2C5"
        plane = NumberPlane(
            x_range=[-2, 5, 1], y_range=[-1, 4, 1],
            x_length=7.35, y_length=5.25, faded_line_ratio=0,
            axis_config={"color": muted, "font_size": 18, "label_constructor": Text},
            x_axis_config={"numbers_to_include": [-1, 1, 2, 3, 4], "label_direction": DR},
            y_axis_config={"numbers_to_include": [1, 2, 3], "label_direction": UL},
            background_line_style={"stroke_color": "#425269", "stroke_width": 1.2},
        ).move_to(2.65 * LEFT + 0.40 * DOWN)
        zero = np.zeros(2)

        def point(xy):
            return plane.c2p(*xy)

        def arrow(start, end, color):
            return Arrow(point(start), point(end), buff=0, color=color,
                         stroke_width=5.5, tip_length=0.21).set_z_index(3)

        def text(value, color=white, size=29):
            return Text(value, font_size=size, color=color)

        def tip_label(name, xy, side, color):
            return text(name, color, 25).next_to(point(xy), side, buff=0.12).set_z_index(5)

        def coords(xy, fmt="g"):
            return f"({xy[0]:{fmt}}, {xy[1]:{fmt}})".replace("-", "−")

        def panel(heading, *rows):
            group = VGroup(text(heading, size=35), *rows)
            group.arrange(DOWN, aligned_edge=LEFT, buff=0.30)
            return group.move_to([1.65, 2.25, 0], aligned_edge=UL)

        # 先画出共用原点的两个输入向量。
        title = text("二维施密特正交化", size=40).to_corner(UL, buff=0.50)
        axes = VGroup(tip_label("x", (5, 0), RIGHT, muted),
                      tip_label("y", (0, 4), UP, muted))
        first, second = arrow(zero, v1, teal), arrow(zero, v2, coral)
        first_label = tip_label("v₁", v1, UP, teal)
        second_label = tip_label("v₂", v2, UP, coral)
        inputs = panel("保留第一个方向",
                       text(f"u₁ = v₁ = {coords(v1)}", teal),
                       text(f"v₂ = {coords(v2)}", coral))
        self.play(FadeIn(title), FadeIn(plane), FadeIn(axes))
        self.play(GrowArrow(first), GrowArrow(second), FadeIn(first_label),
                  FadeIn(second_label), FadeIn(inputs))
        self.wait(1.6)

        # 从 v2 的尖端向 span(u1) 作垂线，原点到垂足的箭头就是 p。
        foot, head = point(projection), point(v2)
        baseline = DashedLine(point(-0.45 * u1), point(1.2 * u1),
                              dash_length=0.12, color=teal, stroke_opacity=0.65)
        drop = DashedLine(head, foot, dash_length=0.10, color=gold, stroke_width=3)
        marker = Dot(head, radius=0.045, color=gold).set_z_index(5)
        foot_angle = RightAngle(Line(foot, point(zero)), Line(foot, head),
                                length=0.22, color=gold, stroke_width=2.5)
        projected_arrow = arrow(zero, projection, gold).set_z_index(4)
        projected_label = tip_label("p", projection, DR, gold)

        # 屏幕上的系数和矩阵元素都取自上面的计算。
        numerator, denominator = text("v₂ · u₁", size=25), text("u₁ · u₁", size=25)
        half_width = (max(numerator.width, denominator.width) + 0.10) / 2
        bar = Line(half_width * LEFT, half_width * RIGHT, stroke_width=2)
        fraction = VGroup(numerator, bar, denominator).arrange(DOWN, buff=0.06)
        formula = VGroup(text("p ="), fraction, text("u₁")).arrange(RIGHT, buff=0.12)
        entries = VGroup(*(text(f"{x:g}", size=25) for x in projector.flat))
        entries.arrange_in_grid(rows=2, cols=2, buff=(0.30, 0.12))
        brackets = [text(ch).stretch_to_fit_height(entries.height + 0.14) for ch in "[]"]
        matrix_row = VGroup(text("P ="), brackets[0], entries, brackets[1]).arrange(RIGHT, buff=0.10)
        projection_panel = panel("投影到 u₁ 所在的直线", formula, matrix_row,
                                 text(f"p = P v₂ = {coefficient:g} u₁ = {coords(projection)}", gold))
        self.play(FadeOut(inputs), run_time=0.35)
        self.play(FadeIn(projection_panel), Create(baseline), FadeIn(marker))
        self.play(Create(drop), marker.animate.move_to(foot), run_time=1.7, rate_func=smooth)
        self.play(GrowArrow(projected_arrow), Create(foot_angle), FadeIn(projected_label))
        self.wait(2.3)

        # 垂足到 v2 的位移就是 v2 - p。平移整支箭头，保持长度和方向。
        remainder = arrow(projection, v2, violet)
        subtraction = panel("减去投影", text("u₂ = v₂ − p", violet),
                            text(f"= {coords(v2)} − {coords(projection)}"),
                            text(f"= {coords(u2)}", violet))
        self.play(FadeOut(projection_panel), run_time=0.35)
        self.play(FadeIn(subtraction), GrowArrow(remainder))
        self.wait(0.9)
        self.play(remainder.animate.shift(point(zero) - foot), run_time=2.1, rate_func=smooth)

        origin_angle = RightAngle(Line(point(zero), point(u1)), Line(point(zero), point(u2)),
                                  length=0.23, color=white, stroke_width=2.5)
        u1_label, u2_label = tip_label("u₁", u1, UP, teal), tip_label("u₂", u2, UL, violet)
        check = text(f"u₁ · u₂ = {u1 @ u2:g}").next_to(subtraction, DOWN, buff=0.38)
        check.align_to(subtraction, LEFT)
        construction = VGroup(second, second_label, baseline, drop, marker, foot_angle,
                              projected_arrow, projected_label)
        self.play(FadeOut(construction), ReplacementTransform(first_label, u1_label),
                  FadeIn(u2_label), Create(origin_angle), FadeIn(check))
        self.wait(2)

        # 此时已经正交。将每个向量单独除以它的长度，得到标准正交基。
        units = panel("将两个向量单位化", text("q₁ = u₁ / ‖u₁‖", teal),
                      text(f"q₁ ≈ {coords(q1, '.3f')}", teal, 25),
                      text("q₂ = u₂ / ‖u₂‖", violet),
                      text(f"q₂ ≈ {coords(q2, '.3f')}", violet, 25),
                      text("‖q₁‖ = ‖q₂‖ = 1"))
        unit_circle = Circle(radius=np.linalg.norm(point((1, 0)) - point(zero)),
                             color=muted, stroke_width=2).move_to(point(zero))
        basis_caption = text("q₁、q₂ 构成标准正交基", size=26)
        basis_caption.next_to(units, DOWN, buff=0.38).align_to(units, LEFT)
        self.play(FadeOut(VGroup(subtraction, check, u1_label, u2_label)), run_time=0.45)
        self.play(FadeIn(units), Create(unit_circle))
        self.play(Transform(first, arrow(zero, q1, teal)),
                  Transform(remainder, arrow(zero, q2, violet)), run_time=1.8)
        self.play(FadeIn(tip_label("q₁", q1, RIGHT, teal)),
                  FadeIn(tip_label("q₂", q2, UL, violet)), FadeIn(basis_caption), run_time=0.55)
        self.wait(2.5)
