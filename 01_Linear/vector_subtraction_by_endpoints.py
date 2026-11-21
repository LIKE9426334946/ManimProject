"""在仓库根目录运行 `uv run manim -pqm 01_Linear/vector_subtraction_by_endpoints.py VectorSubtractionByEndpoints`。"""

import numpy as np
from manim import (
    DL, DOWN, LEFT, RIGHT, UL, UP, Arrow, Create, DashedLine, DecimalNumber,
    Dot, FadeIn, FadeOut, NumberPlane, Scene, Text, VGroup, smooth,
)


# 第一项是起始向量，后续各项按顺序减去。
VECTORS = [
    ("a", (4, 2)),
    ("b", (1, 3)),
    ("c", (1, -3)),
    ("d", (4, 1)),
]


class VectorSubtractionByEndpoints(Scene):
    def construct(self):
        if len(VECTORS) < 2:
            raise ValueError("请设置一个起始向量和至少一个要减去的向量。")
        names, coordinates = zip(*VECTORS)
        vectors = np.asarray(coordinates, dtype=float)
        if vectors.shape != (len(names), 2) or not np.isfinite(vectors).all():
            raise ValueError("每个向量需要两个有限的坐标值。")

        partials = [vectors[0]]
        for operand in vectors[1:]:
            partials.append(partials[-1] - operand)

        self.camera.background_color = "#101722"
        teal, coral, gold = "#64D8D0", "#FF977D", "#FFD076"
        white, muted = "#EEF3FA", "#A4B2C5"
        bounds = np.vstack(([0, 0], vectors, partials))
        low = np.floor(bounds.min(axis=0) - 0.5).astype(int)
        high = np.ceil(bounds.max(axis=0) + 0.5).astype(int)
        width, height = high - low
        unit = min(7.7 / width, 5.85 / height)
        tick = max(1, int(np.ceil(max(width, height) / 12)))
        plane = NumberPlane(
            x_range=[low[0], high[0], tick], y_range=[low[1], high[1], tick],
            x_length=width * unit, y_length=height * unit, faded_line_ratio=0,
            axis_config={"color": muted, "include_numbers": True, "font_size": 17,
                         "label_constructor": Text},
            x_axis_config={"label_direction": DOWN, "numbers_to_exclude": [0, high[0]]},
            y_axis_config={"label_direction": LEFT, "numbers_to_exclude": [0, high[1]]},
            background_line_style={"stroke_color": "#425269", "stroke_width": 1.2},
        ).move_to(2.95 * LEFT + 0.45 * DOWN)
        origin = plane.c2p(0, 0)

        def text(value, color=white, size=28):
            return Text(value, color=color, font_size=size)

        def coords(xy):
            return f"({xy[0]:g}, {xy[1]:g})".replace("-", "−")

        def sidebar(mob, y):
            if mob.width > 5.45:
                mob.scale_to_fit_width(5.45)
            return mob.move_to([1.00, y, 0], aligned_edge=LEFT)

        def vector_mark(start, end, color):
            if np.allclose(start, end, atol=1e-10, rtol=0):
                return Dot(start, radius=0.063, color=color)
            return Arrow(start, end, buff=0, color=color, stroke_width=5.5, tip_length=0.20)

        def tip_label(name, tip, side, color):
            return text(name, color, 25).next_to(tip, side, buff=0.12).set_z_index(6)

        def endpoint_table(tail_point, head_point):
            # 直接读取金色箭头两端的实际位置，换算回图上的坐标。
            group = VGroup()
            for col, (letter, x) in enumerate((("x", 3.30), ("y", 5.05))):
                group.add(text(letter, muted, 21).move_to([x, -1.00, 0], aligned_edge=RIGHT))
                for y, position in ((-1.54, tail_point), (-2.12, head_point)):
                    number = DecimalNumber(plane.p2c(position())[col], mob_class=Text, font_size=29,
                                           num_decimal_places=1, color=gold, edge_to_fix=RIGHT)
                    number.move_to([x, y, 0], aligned_edge=RIGHT)
                    number.add_updater(lambda m, get=position, j=col: m.set_value(
                        round(float(plane.p2c(get())[j]), 1) or 0.0))
                    group.add(number)
            group.add(sidebar(text("箭尾", muted, 24), -1.54), sidebar(text("箭尖", muted, 24), -2.12))
            return group

        title = text("用终点连线做向量连减", size=38).to_corner(UL, buff=0.5)
        axes = VGroup(
            text("x", muted, 22).next_to(plane.c2p(high[0], 0), RIGHT, buff=0.14),
            text("y", muted, 22).next_to(plane.c2p(0, high[1]), RIGHT, buff=0.12),
            text("O", muted, 19).next_to(origin, DL, buff=0.10),
        )
        expression = sidebar(text("S = " + " − ".join(names), size=34), 2.65)
        current_name = names[0]
        current_tip = plane.c2p(*partials[0])
        current = vector_mark(origin, current_tip, teal).set_z_index(3)
        current_label = tip_label(current_name, current_tip, UP, teal)
        self.play(FadeIn(title), FadeIn(plane), FadeIn(axes), FadeIn(expression))
        self.play(FadeIn(current), FadeIn(current_label), run_time=0.65)

        summary = VGroup()
        subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        for i in range(1, len(vectors)):
            name, operand = names[i], vectors[i]
            before, result = partials[i - 1], partials[i]
            result_name = "r" + str(i).translate(subscripts)
            operand_tip, result_tip = plane.c2p(*operand), plane.c2p(*result)
            other = vector_mark(origin, operand_tip, coral).set_z_index(3)
            other_label = tip_label(name, operand_tip, DOWN + RIGHT, coral)
            operation = sidebar(text(f"{result_name} = {current_name} − {name}", size=32), 1.83)
            inputs = VGroup(text(f"{current_name} = {coords(before)}", teal, 25),
                            text(f"{name} = {coords(operand)}", coral, 25)).arrange(RIGHT, buff=0.36)
            sidebar(inputs, 1.22)
            self.play(current.animate.set_color(teal), current_label.animate.set_color(teal),
                      FadeIn(other), FadeIn(other_label), FadeIn(operation), FadeIn(inputs), run_time=0.85)
            self.wait(0.65)

            # 对 r - v，先从 v 的终点画到 r 的终点。金色小圆点标出箭尾。
            difference = vector_mark(operand_tip, current_tip, gold).set_z_index(5)
            tail = Dot(operand_tip, radius=0.052, color=gold).set_z_index(7)
            zero = isinstance(difference, Dot)
            action = "两个终点重合，得到零向量" if zero else f"从 {name} 的终点指向 {current_name} 的终点"
            connect = sidebar(text(action, size=27), 0.45)
            invariant = VGroup(text(f"{result_name} = {coords(result)}", gold),
                               text(f"长度 ≈ {np.linalg.norm(result):.2f}", muted, 23)).arrange(RIGHT, buff=0.38)
            sidebar(invariant, -0.16)
            table = endpoint_table(tail.get_center, difference.get_center if zero else difference.get_end)
            self.play(FadeIn(difference), FadeIn(tail), FadeIn(connect), FadeIn(invariant), FadeIn(table))
            self.wait(1.1)

            guides = VGroup()
            if np.any(operand):
                offset = origin - operand_tip
                reference = difference.copy().set_opacity(0.28).set_z_index(2)
                trail = DashedLine(current_tip, result_tip, dash_length=0.10, color=muted, stroke_width=2)
                guides.add(reference, trail)
                self.add(reference)
                slide = sidebar(text(f"将箭尾从 {name} 的终点平移到原点", size=27), 0.45)
                self.play(FadeOut(connect), FadeIn(slide), current.animate.set_opacity(0.40),
                          other.animate.set_opacity(0.52), run_time=0.40)
                # 两端同时加上同一个位移 -v。箭尾 v 到达 O，箭尖 r 到达 r - v。
                self.play(difference.animate.shift(offset), tail.animate.shift(offset),
                          Create(trail), run_time=2.35, rate_func=smooth)
            else:
                slide = sidebar(text("箭尾已经在原点", size=27), 0.45)
                self.play(FadeOut(connect), FadeIn(slide), run_time=0.40)
            self.wait(0.95)

            landed_label = tip_label(result_name, result_tip, UP, gold)
            self.play(FadeOut(VGroup(current, current_label, other, other_label, guides, tail)),
                      FadeIn(landed_label), run_time=0.50)
            self.play(FadeOut(VGroup(operation, inputs, slide, invariant, table)), run_time=0.35)
            summary.add(text(f"{result_name} = {current_name} − {name} = {coords(result)}", size=27))
            current_name, current_tip = result_name, result_tip
            current, current_label = difference, landed_label

        summary.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        if summary.height > 3.20:
            summary.scale_to_fit_height(3.20)
        sidebar(summary, 0.77)
        answer = sidebar(text(f"S = {coords(partials[-1])}", gold, 34), -1.85)
        self.play(FadeIn(summary), FadeIn(answer), run_time=0.65)
        self.wait(2.4)
