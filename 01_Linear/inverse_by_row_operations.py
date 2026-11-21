"""在仓库根目录运行 uv run manim -pqm 01_Linear/inverse_by_row_operations.py InverseByRowOperations。"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from manim import (
    DOWN, LEFT, RIGHT, UP, AnimationGroup, Arrow, DashedVMobject, Dot, FadeIn,
    FadeOut, Line, NumberPlane, Polygon, Rectangle, ReplacementTransform, Scene,
    Succession, Text, Transform, VGroup, VMobject, Wait,
)


# 可改为其他 2×2 矩阵。分数也可以写成字符串，例如 "1/3"。
MATRIX = ((1, 2), (-1, 1))
IDENTITY = np.array([[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]])
INK, MUTED, GRID = "#EEF3FA", "#A6B6CA", "#354359"
TEAL, CORAL, GOLD = "#64D8D0", "#FF977D", "#FFD076"
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def scalar(value):
    return str(value).replace("-", "−")


def linear_form(coefficients, variables):
    """用有理系数排出行操作和坐标规则，如 R₁ − 2R₂、y/3。"""
    terms = []
    for coefficient, variable in zip(coefficients, variables):
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        term = (str(magnitude.numerator) if magnitude.numerator != 1 else "") + variable
        if magnitude.denominator != 1:
            term += f"/{magnitude.denominator}"
        sign = (" − " if terms else "−") if coefficient < 0 else (" + " if terms else "")
        terms.append(sign + term)
    return "".join(terms) or "0"


@dataclass
class Step:
    elementary: np.ndarray
    augmented: np.ndarray
    rows: tuple[int, ...]
    operation: str
    purpose: str


def row_reduce(matrix):
    """每一步都用 E 左乘整个 [L | R]，因此始终有 L = R @ A。"""
    if np.shape(matrix) != (2, 2):
        raise ValueError("请给出两行两列的矩阵。")
    a = np.array([[Fraction(str(n)) for n in row] for row in matrix])
    initial = np.hstack((a, IDENTITY))
    state, steps = initial, []
    names = ("R₁", "R₂")

    def apply(elementary, rows, operation, purpose):
        nonlocal state
        state = elementary @ state
        steps.append(Step(elementary, state, rows, operation, purpose))

    for pivot in range(2):
        source = next((r for r in range(pivot, 2) if state[r, pivot] != 0), None)
        if source is None:
            return initial, steps, pivot
        if source != pivot:
            apply(IDENTITY[::-1].copy(), (0, 1), "R₁ ↔ R₂", "交换两行，取得非零主元")
        if state[pivot, pivot] != 1:
            e = IDENTITY.copy()
            e[pivot, pivot] = 1 / state[pivot, pivot]
            operation = linear_form((e[pivot, pivot],), (names[pivot],))
            apply(e, (pivot,), f"{names[pivot]} ← {operation}", f"把第 {pivot + 1} 列的主元化为 1")
        other = 1 - pivot
        if state[other, pivot] != 0:
            e = IDENTITY.copy()
            e[other, pivot] = -state[other, pivot]
            operation = linear_form((Fraction(1), e[other, pivot]), (names[other], names[pivot]))
            purpose = f"把第 {pivot + 1} 列的 {scalar(state[other, pivot])} 消为 0"
            apply(e, (other,), f"{names[other]} ← {operation}", purpose)
    return initial, steps, None


def label(value, size=28, color=INK):
    return Text(value, font_size=size, color=color)


def place(mob, x, y, width):
    if mob.width > width:
        mob.scale_to_fit_width(width)
    return mob.move_to([x, y, 0], aligned_edge=LEFT)


def entry(value, color):
    if value.denominator == 1:
        mob = label(scalar(value), 36, color)
    else:
        top = label(scalar(value.numerator), 23, color)
        bottom = label(str(value.denominator), 23, color)
        half = (max(top.width, bottom.width) + 0.10) / 2
        bar = Line(half * LEFT, half * RIGHT, color=color, stroke_width=2)
        mob = VGroup(top, bar, bottom).arrange(DOWN, buff=0.040)
    if mob.width > 0.92:
        mob.scale_to_fit_width(0.92)
    return mob


def matrix_picture(values, augmented=False, color=INK):
    """括号和数字分开，行变换时只更新实际改变的格子。"""
    ncols = values.shape[1]
    columns = (np.arange(ncols) - (ncols - 1) / 2) * 1.12
    edge = columns[-1] + 0.54
    frame = VGroup()
    for side in (-1, 1):
        outside, inside = side * edge, side * (edge - 0.14)
        bracket = VMobject(stroke_color=MUTED, stroke_width=3)
        bracket.set_points_as_corners([[inside, 0.91, 0], [outside, 0.91, 0],
                                       [outside, -0.91, 0], [inside, -0.91, 0]])
        frame.add(bracket)
    if augmented:
        frame.add(Line(0.84 * DOWN, 0.84 * UP, color=MUTED, stroke_width=1.8))
    cells = VGroup()
    for (r, c), value in np.ndenumerate(values):
        ink = (TEAL, CORAL, GOLD, GOLD)[c] if augmented else color
        cells.add(entry(value, ink).move_to([columns[c], 0.45 - 0.90 * r, 0]))
    return VGroup(frame, cells)


class InverseByRowOperations(Scene):
    def construct(self):
        self.camera.background_color = "#101722"
        start, steps, missing = row_reduce(MATRIX)
        a = start[:, :2]

        # 网格保持固定。按各步左块两列及其和选取范围，保证平行四边形始终入镜。
        blocks = [IDENTITY, a, *(step.augmented[:, :2] for step in steps)]
        corners = np.vstack([np.zeros(2)] + [
            p for block in blocks for p in (*block.T, block.sum(axis=1))
        ]).astype(float)
        low, high = corners.min(axis=0) - 0.62, corners.max(axis=0) + 0.62
        box = np.array([6.05, 4.65])
        span = box / min(box / (high - low))
        low, high = (low + high - span) / 2, (low + high + span) / 2
        tick = max(1, int(np.ceil(max(span) / 9)))
        plane = NumberPlane(
            x_range=[low[0], high[0], tick], y_range=[low[1], high[1], tick],
            x_length=box[0], y_length=box[1], faded_line_ratio=0,
            axis_config={"color": MUTED, "stroke_width": 1.7}, tips=False,
            background_line_style={"stroke_color": GRID, "stroke_width": 1.2},
        ).move_to([-3.48, -0.20, 0])
        origin = plane.c2p(0, 0)
        axes = VGroup(label("O", 16, MUTED).next_to(origin, DOWN + LEFT, buff=0.055))
        for axis, side, name in ((0, DOWN, "x"), (1, LEFT, "y")):
            for k in range(int(np.ceil((low[axis] + 0.12) / tick)),
                           int(np.floor((high[axis] - 0.12) / tick)) + 1):
                if k:
                    pos = np.zeros(2)
                    pos[axis] = k * tick
                    axes.add(label(scalar(k * tick), 16, MUTED).next_to(plane.c2p(*pos), side, buff=0.06))
            pos = np.zeros(2)
            pos[axis] = high[axis]
            axes.add(label(name, 20, MUTED).next_to(plane.c2p(*pos), RIGHT, buff=0.07))
        axes.set_z_index(4)

        def footprint(block):
            v, w = np.array(block, dtype=float).T
            p, q, corner = (plane.c2p(*xy) for xy in (v, w, v + w))
            shape = Polygon(origin, p, corner, q, color="#8CBAC8", stroke_width=2.2,
                            fill_color=TEAL, fill_opacity=0.14).set_z_index(1)
            arrows = [
                (Dot(origin, radius=0.045, color=c) if np.array_equal(origin, tip) else
                 Arrow(origin, tip, buff=0, color=c, stroke_width=5, tip_length=0.19)).set_z_index(3)
                for tip, c in ((p, TEAL), (q, CORAL))
            ]
            return VGroup(shape, *arrows, Dot(corner, radius=0.055, color=INK).set_z_index(4))

        def change_shape(old, new, transform):
            # 交换坐标或负倍缩放会翻转定向，直接淡入镜像位置。
            # 这样不会把过渡中出现的压扁状态误当作一次行操作。
            determinant = transform[0, 0] * transform[1, 1] - transform[0, 1] * transform[1, 0]
            if determinant <= 0:
                return AnimationGroup(FadeOut(old), FadeIn(new))
            return ReplacementTransform(old, new)

        def caption(heading, formula):
            return VGroup(place(label(heading, 21, MUTED), -6.51, -2.94, 6.08),
                          place(label(formula, 30), -6.51, -3.48, 6.08))

        def detail(heading, formula, bottom):
            return VGroup(place(label(heading, 24, MUTED), 0.60, -0.66, 6.03),
                          place(label(formula, 36, GOLD), 0.60, -1.56, 6.03),
                          place(bottom, 0.60, -2.91, 6.03))

        title = place(label("用初等行变换求逆矩阵", 39), -6.55, 3.40, 7.50)
        goal = label("[ A | I ]   →   [ I | A⁻¹ ]", 29).move_to([3.48, 3.40, 0])
        rule = Line([-6.55, 2.96, 0], [6.55, 2.96, 0], color=GRID, stroke_width=1.5)
        legend = VGroup(label("左块第 1 列", 23, TEAL), label("第 2 列", 23, CORAL),
                        label("虚线为单位正方形", 20, MUTED)).arrange(RIGHT, buff=0.30)
        place(legend, -6.50, 2.54, 6.05)
        headings = VGroup(label("左块", 24).move_to([2.36, 2.54, 0]),
                          label("右块", 24, GOLD).move_to([4.60, 2.54, 0]))
        board = matrix_picture(start, augmented=True).move_to([3.48, 1.13, 0])
        shape = footprint(IDENTITY)
        standard = DashedVMobject(shape[0].copy().set_fill(opacity=0).set_stroke(
            MUTED, width=2.5), num_dashes=25, dashed_ratio=0.58).set_z_index(2)
        corner_image = ", ".join(scalar(n) for n in a.sum(axis=1))
        note = caption("白点是单位正方形右上角的像", f"A 把 (1, 1) 送到 ({corner_image})")
        card = detail("构造增广矩阵", "在 A 右边添上 I", label("对左右两块同时做相同的行变换。", 24))
        self.play(FadeIn(VGroup(title, goal, rule, legend, headings, plane, axes, standard)),
                  FadeIn(board), FadeIn(shape), run_time=1.0)
        self.play(FadeIn(note), FadeIn(card), run_time=0.55)
        transformed = footprint(a)
        self.play(change_shape(shape, transformed, a), run_time=1.65)
        shape, previous = transformed, start
        self.wait(1.35)

        for idx, step in enumerate(steps, start=1):
            sub = str(idx).translate(SUBSCRIPT)
            elementary = VGroup(label(f"本步左乘  E{sub} =", 25),
                                matrix_picture(step.elementary).scale(0.63)).arrange(RIGHT, buff=0.22)
            next_card = detail(f"第 {idx} 步  ·  {step.purpose}", step.operation, elementary)
            x_rule, y_rule = (linear_form(row, ("x", "y")) for row in step.elementary)
            next_note = caption("平面上每个点同步移动", f"(x, y)  →  ({x_rule}, {y_rule})")
            highlight = VGroup(*[
                Rectangle(width=board.width + 0.16, height=0.83, stroke_width=1.3, color=GOLD,
                          fill_opacity=0.07).move_to([3.48, board[1][4*r].get_y(), 0]).set_z_index(-1)
                for r in step.rows
            ])
            self.play(FadeOut(card), FadeOut(note), run_time=0.32)
            self.play(FadeIn(next_card), FadeIn(next_note), FadeIn(highlight), run_time=0.50)
            self.wait(1.05)
            target = matrix_picture(step.augmented, augmented=True).move_to(board)
            updates = [Transform(board[1][j], target[1][j]) for j, (before, after)
                       in enumerate(zip(previous.flat, step.augmented.flat)) if before != after]
            transformed = footprint(step.augmented[:, :2])
            self.play(change_shape(shape, transformed, step.elementary),
                      Succession(Wait(0.45), AnimationGroup(*updates), Wait(0.30)), run_time=1.80)
            self.wait(1.45)
            self.play(FadeOut(highlight), run_time=0.26)
            shape, previous, card, note = transformed, step.augmented, next_card, next_note

        self.play(FadeOut(card), FadeOut(note), run_time=0.35)
        if missing is None:
            inverse = previous[:, 2:]
            product = VGroup(label("验算", 22, MUTED), matrix_picture(inverse, color=GOLD).scale(0.57),
                             label("×", 24), matrix_picture(a).scale(0.57), label("=", 24),
                             matrix_picture(inverse @ a).scale(0.57)).arrange(RIGHT, buff=0.16)
            factors = " ".join("E" + str(i).translate(SUBSCRIPT) for i in range(len(steps), 0, -1)) or "I"
            note = caption("两列分别回到标准基 e₁、e₂", f"A⁻¹ 把 ({corner_image}) 送回 (1, 1)")
            result_labels = VGroup(label("左块  I", 24).move_to(headings[0]),
                                   label("右块  A⁻¹", 24, GOLD).move_to(headings[1]))
            reason = detail("把所有行变换合起来，记为 E", "EA = I   ⇒   E = A⁻¹",
                            label("右块从 I 出发，最后得到 EI = E。", 24))
            self.play(Transform(headings, result_labels), FadeIn(reason), FadeIn(note), run_time=0.8)
            self.wait(2.65)
            self.play(FadeOut(reason), run_time=0.30)
            card = detail("左块化成 I，右块就是逆矩阵", f"A⁻¹ = {factors}", product)
            self.play(FadeIn(card), run_time=0.55)
        else:
            card = detail(f"第 {missing + 1} 列找不到可用的主元", "这个矩阵没有逆矩阵",
                          label("左块无法通过行变换化成 I。", 25))
            note = caption("图形的面积为零", "两个列向量线性相关")
            self.play(FadeIn(card), FadeIn(note), run_time=0.8)
        self.wait(3.0)
