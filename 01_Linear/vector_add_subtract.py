"""在仓库根目录运行 `uv run manim -pqm 01_Linear/vector_add_subtract.py VectorAddSubtract`。"""

import numpy as np
from manim import (
    DL, DOWN, LEFT, PI, RIGHT, UL, UP, Arc, Arrow, Create, DecimalNumber,
    Dot, FadeIn, FadeOut, GrowArrow, NumberPlane, Rotate, Scene, Text, VGroup,
    always_redraw, smooth,
)


# 按列表顺序计算。每项依次是符号（+1 或 -1）、向量名、原始坐标和颜色。
STEPS = [
    (+1, "a", (3, 1), "#64D8D0"),
    (+1, "b", (1, 2), "#FF977D"),
    (-1, "c", (2, -1), "#B6A1FF"),
    (+1, "d", (-3, -2), "#76BBFF"),
]


class VectorAddSubtract(Scene):
    def construct(self):
        if not STEPS or any(sign not in (-1, 1) for sign, *_ in STEPS):
            raise ValueError("请至少设置一个向量，每项的符号为 +1 或 -1。")
        signs = np.array([sign for sign, *_ in STEPS])
        vectors = np.array([xy for _, _, xy, _ in STEPS], dtype=float)
        moves = signs[:, None] * vectors
        stops = np.vstack((np.zeros(2), np.cumsum(moves, axis=0)))

        # 同时容纳原始向量、反向时的半圆和各步终点，整个动画固定比例。
        turn_radius = np.linalg.norm(vectors[signs < 0], axis=1).max(initial=0)
        bounds = np.vstack((vectors, moves, stops, [-turn_radius] * 2, [turn_radius] * 2))
        low = np.floor(bounds.min(axis=0) - 0.4).astype(int)
        high = np.ceil(bounds.max(axis=0) + 0.4).astype(int)
        width, height = high - low
        unit = min(7.7 / width, 5.95 / height)
        tick = max(1, int(np.ceil(max(width, height) / 12)))

        self.camera.background_color = "#101722"
        white, muted, gold = "#EEF3FA", "#A4B2C5", "#FFD076"
        plane = NumberPlane(
            x_range=[low[0], high[0], tick], y_range=[low[1], high[1], tick],
            x_length=unit * width, y_length=unit * height, faded_line_ratio=0,
            axis_config={"color": muted, "include_numbers": True,
                         "font_size": 17, "label_constructor": Text},
            x_axis_config={"label_direction": DOWN, "numbers_to_exclude": [0, high[0]]},
            y_axis_config={"label_direction": LEFT, "numbers_to_exclude": [0, high[1]]},
            background_line_style={"stroke_color": "#425269", "stroke_width": 1.2},
        ).move_to(2.90 * LEFT + 0.45 * DOWN)
        origin = plane.c2p(0, 0)

        def text(value, color=white, size=28):
            return Text(value, color=color, font_size=size)

        def coords(xy):
            return f"({xy[0]:g}, {xy[1]:g})".replace("-", "−")

        def arrow(start, end, color, width=5.5, tip=0.20):
            return Arrow(start, end, buff=0, color=color, stroke_width=width, tip_length=tip)

        title = text("二维向量的连续加减", size=39).to_corner(UL, buff=0.50)
        axes = VGroup(
            text("x", muted, 23).next_to(plane.c2p(high[0], 0), RIGHT, buff=0.19),
            text("y", muted, 23).next_to(plane.c2p(0, high[1]), RIGHT, buff=0.12),
            text("O", muted, 18).next_to(origin, DL, buff=0.09),
        )
        expression = VGroup(text("S =", size=34))
        for i, (sign, name, _, color) in enumerate(STEPS):
            prefix = "− " if sign < 0 else ("+ " if i else "")
            expression.add(text(prefix + name, color, 34))
        expression.arrange(RIGHT, buff=0.15)
        if expression.width > 5.35:
            expression.scale_to_fit_width(5.35)
        expression.move_to([1.20, 2.26, 0], aligned_edge=LEFT)

        description = VGroup()

        def explain(action, calculation, color):
            nonlocal description
            self.remove(description)
            description = VGroup(text(action, size=29), text(calculation, color, 28))
            description.arrange(DOWN, aligned_edge=LEFT, buff=0.24)
            if description.width > 5.35:
                description.scale_to_fit_width(5.35)
            description.move_to([1.20, 1.30, 0], aligned_edge=UL)
            self.add(description)

        # 金点沿折线行走。金色合向量永远从原点指向它，经过原点时长度为零。
        marker = Dot(origin, radius=0.057, color=gold).set_z_index(7)
        resultant = always_redraw(lambda: arrow(origin, marker.get_center(), gold, width=3, tip=0.14))
        resultant.set_z_index(5)
        readout = VGroup(text("当前终点坐标", muted, 25).move_to([1.20, -1.03, 0], aligned_edge=LEFT))
        for axis, name in enumerate(("x", "y")):
            label = text(f"{name} =", size=29).move_to([1.20 + 2.45 * axis, -1.68, 0], aligned_edge=LEFT)
            number = DecimalNumber(0, num_decimal_places=1, mob_class=Text, font_size=34, color=gold)
            number.next_to(label, RIGHT, buff=0.15)
            number.add_updater(lambda m, i=axis: m.set_value(plane.p2c(marker.get_center())[i]))
            readout.add(label, number)

        # 先从同一原点展示所有输入，随后逐项把箭头接成一条折线。
        sources = VGroup()
        for _, name, xy, color in STEPS:
            head = plane.c2p(*xy)
            sources.add(arrow(origin, head, color),
                        text(name, color, 24).next_to(head, UP if xy[1] >= 0 else DOWN, buff=0.10))
        self.play(FadeIn(title), FadeIn(plane), FadeIn(axes), FadeIn(expression))
        self.play(FadeIn(sources), FadeIn(marker), FadeIn(readout))
        self.wait(1)
        self.play(FadeOut(sources), run_time=0.45)
        self.add(resultant)

        for i, (sign, name, _, color) in enumerate(STEPS):
            vector, move = vectors[i], moves[i]
            start, end = stops[i], stops[i + 1]
            start_point, end_point = plane.c2p(*start), plane.c2p(*end)
            signed_name = name if sign > 0 else f"−{name}"
            symbol = "+" if sign > 0 else "−"
            equation = f"{coords(start)} {symbol} {coords(vector)} = {coords(end)}"
            for j, token in enumerate(expression[1:]):
                token.set_opacity(1 if j <= i else 0.38)

            if not np.any(vector):
                explain(f"{name} 是零向量，终点停在原处", equation, color)
                self.wait(1.5)
                continue
            explain(f"从原点画出 {name}", f"{name} = {coords(vector)}", color)
            moving = arrow(origin, plane.c2p(*vector), color).set_z_index(4)
            self.play(GrowArrow(moving), run_time=0.70)

            if sign < 0:
                # 旋转整支箭头，保持长度。圆弧是尖端走过的轨迹。
                explain(f"绕原点转 180°，得到 {signed_name}", f"{signed_name} = {coords(move)}", color)
                ghost = moving.copy().set_opacity(0.24).set_z_index(1)
                arc = Arc(radius=np.linalg.norm(plane.c2p(*vector) - origin),
                          start_angle=np.arctan2(vector[1], vector[0]), angle=PI,
                          arc_center=origin, color=color, stroke_width=2, stroke_opacity=0.6)
                self.add(ghost)
                self.play(Rotate(moving, PI, about_point=origin), Create(arc), run_time=1.7, rate_func=smooth)
                self.wait(0.55)
                self.play(FadeOut(ghost), FadeOut(arc), run_time=0.30)

            if not np.allclose(start, 0):
                explain(f"将箭尾平移到 {coords(start)}", f"{signed_name} = {coords(move)}", color)
                reference = moving.copy().set_opacity(0.23).set_z_index(1)
                self.add(reference)
                self.play(moving.animate.shift(start_point - origin), run_time=1.65, rate_func=smooth)
                self.remove(reference)

            # 保留首尾相接的箭头，让金点沿它移动，实时更新合向量。
            normal = np.array([move[1], -move[0], 0]) / np.linalg.norm(move)
            tag = text(signed_name, color, 25).move_to((start_point + end_point) / 2 + 0.26 * normal)
            tag.set_z_index(5)
            explain(f"沿 {signed_name} 移动", equation, color)
            self.play(FadeIn(tag), marker.animate.move_to(end_point), run_time=1.55, rate_func=smooth)
            self.wait(0.75)

        explain("合向量指向折线的终点", f"S = {coords(stops[-1])}", gold)
        final_label = text("S", gold, 25).next_to(marker, UL, buff=0.11).set_z_index(6)
        self.play(FadeIn(final_label), run_time=0.50)
        self.wait(2.5)
