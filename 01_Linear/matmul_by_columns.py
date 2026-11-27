"""在仓库根目录运行 uv run manim -pqm 01_Linear/matmul_by_columns.py MatmulByColumns。"""

from fractions import Fraction

import numpy as np
from manim import (
    RIGHT, UP, AnimationGroup, FadeIn, FadeOut, LaggedStart, Line,
    ReplacementTransform, RoundedRectangle, Scene, Transform, TransformFromCopy,
    VGroup, VMobject,
)

from inverse_by_row_operations import CORAL, GOLD, GRID, INK, MUTED, SUBSCRIPT, TEAL, label, place, scalar


# x 的三列与 W 的三行按顺序配对。可修改整数、小数或用 "1/2" 这样的分数。
X_VALUES = ((1, 2, 3), (4, 5, 6))
W_VALUES = ((2, 1), (3, -1), (4, 2))
COLORS = (TEAL, CORAL, "#BBA9FF")
TERM_X = (-5.03, -1.76, 1.51)
WORK_Y = -1.65
ANSWER_X = 5.25


def as_matrix(values, shape):
    if np.shape(values) != shape:
        raise ValueError(f"矩阵应为 {shape[0]} 行 {shape[1]} 列。")
    return np.array([[Fraction(str(v)) for v in row] for row in values], dtype=object)


def symbols(name, rows, cols):
    return np.array([[name + f"{r + 1}{c + 1}".translate(SUBSCRIPT)
                      for c in range(cols)] for r in range(rows)])


class DisplayMatrix(VGroup):
    """独立定位括号和每个格子，以便复制某一列或单个权重。"""

    def __init__(self, values, colors=INK, dx=1.03, dy=0.69, size=31):
        rows, cols = len(values), len(values[0])
        self.ncols, self.dx, self.dy = cols, dx, dy
        palette = np.broadcast_to(colors, (rows, cols))
        self.cells = VGroup()
        for r, row in enumerate(values):
            for c, value in enumerate(row):
                glyph = value if isinstance(value, VMobject) else label(scalar(value), size, str(palette[r, c]))
                if glyph.width > dx - 0.16:
                    glyph.scale_to_fit_width(dx - 0.16)
                glyph.move_to([(c - (cols - 1) / 2) * dx, ((rows - 1) / 2 - r) * dy, 0])
                self.cells.add(glyph)
        half_width, half_height = cols * dx / 2, (rows - 1) * dy / 2 + 0.29
        self.brackets = VGroup()
        for sign in (-1, 1):
            edge, inner = sign * half_width, sign * (half_width - 0.13)
            bracket = VMobject(stroke_color=MUTED, stroke_width=2.6)
            bracket.set_points_as_corners([[inner, half_height, 0], [edge, half_height, 0],
                                           [edge, -half_height, 0], [inner, -half_height, 0]])
            self.brackets.add(bracket)
        super().__init__(self.brackets, self.cells)

    def at(self, row, col):
        return self.cells[row * self.ncols + col]

    def column(self, col):
        return VGroup(*self.cells.submobjects[col::self.ncols])

    def highlight(self, col, first, last, color, outer=False):
        p, q = self.at(first, col).get_center(), self.at(last, col).get_center()
        pad = 0.12 if outer else -0.10
        return RoundedRectangle(width=self.dx + pad, height=abs(p[1] - q[1]) + 0.54 + pad,
                                corner_radius=0.09, color=color, stroke_width=1.8,
                                fill_opacity=0.06).move_to((p + q) / 2).set_z_index(-1)


def scaled_term(weight, column, k):
    factor_text = scalar(weight)
    if factor_text.startswith("−"):
        factor_text = f"({factor_text})"
    factor = label(factor_text, 31, COLORS[k])
    vector = DisplayMatrix([[v] for v in column], COLORS[k], dx=1.04)
    group = VGroup(factor, label("×", 25, MUTED), vector).arrange(RIGHT, buff=0.13)
    if group.width > 2.80:
        group.scale_to_fit_width(2.80)
    return group.move_to([TERM_X[k], WORK_Y, 0])


def column_vector(values, x, color, dx=1.05):
    return DisplayMatrix([[v] for v in values], color, dx=dx).move_to([x, WORK_Y, 0])


def note(heading, content, x):
    return VGroup(place(label(heading, 21, MUTED), x, -3.00, 6.0),
                  place(content, x, -3.55, 6.0))


def row_sum_line(values, y_name):
    pieces = VGroup(label(y_name + " =", 27))
    for k, v in enumerate(values):
        if k:
            pieces.add(label("−" if v < 0 else "+", 24, MUTED))
        pieces.add(label(scalar(abs(v) if k else v), 27, COLORS[k]))
    pieces.add(label("=", 24, MUTED), label(scalar(sum(values)), 27, GOLD))
    return pieces.arrange(RIGHT, buff=0.13)


class MatmulByColumns(Scene):
    def construct(self):
        self.camera.background_color = "#101722"
        x, w = as_matrix(X_VALUES, (2, 3)), as_matrix(W_VALUES, (3, 2))
        y = x @ w
        x_names, w_names, y_names = symbols("x", 2, 3), symbols("W", 3, 2), symbols("Y", 2, 2)
        x_boards = [DisplayMatrix(a, COLORS, dx=1.16).move_to([-4.62, 1.44, 0]) for a in (x_names, x)]
        w_boards = [DisplayMatrix(a, np.array(COLORS)[:, None], dx=1.14, dy=0.62).move_to([0, 1.44, 0])
                    for a in (w_names, w)]
        targets = DisplayMatrix(y_names, MUTED, dx=1.17).move_to([4.66, 1.44, 0])
        answers = DisplayMatrix(y, GOLD, dx=1.17).move_to(targets)
        for board in w_boards:
            board.column(1).set_opacity(0.42)
        title = place(label("矩阵乘法按列计算", 39), -6.52, 3.48, 7.30)
        self.status = place(label("计算 Y 的第 1 列", 26, GOLD), 3.10, 3.48, 3.48)
        headers = VGroup()
        for name, shape, xpos in (("x", "2 行 3 列", -4.62), ("W", "3 行 2 列", 0), ("Y", "2 行 2 列", 4.66)):
            head = VGroup(label(name, 29), label(shape, 20, MUTED)).arrange(RIGHT, buff=0.22)
            headers.add(head.move_to([xpos, 2.70, 0]))
        rules = VGroup(*(Line([-6.54, h, 0], [6.54, h, 0], color=GRID, stroke_width=1.4)
                         for h in (3.06, 0.34, -2.62)))
        operators = VGroup(label("×", 38).move_to([-2.03, 1.44, 0]), label("=", 38).move_to([2.34, 1.44, 0]))
        self.hint = self.hint_text("取 W 的第 1 列，从上往下读三个权重")
        memory = VGroup(note("W 的第一个下标 k", label("对应 x 的第 k 列", 28), -6.52),
                        note("W 的第二个下标 j", label("对应 Y 的第 j 列", 28), 0.56))
        focus_w = w_boards[0].highlight(0, 0, 2, GOLD, outer=True)
        focus_y = targets.highlight(0, 0, 1, GOLD, outer=True)
        self.play(FadeIn(VGroup(title, self.status, headers, rules, operators)),
                  FadeIn(x_boards[0]), FadeIn(w_boards[0]), FadeIn(targets), run_time=0.9)
        self.play(FadeIn(self.hint), FadeIn(memory), FadeIn(focus_w), FadeIn(focus_y), run_time=0.55)
        self.wait(1.2)

        for j in range(2):
            if j == 1:
                self.play(FadeOut(work), FadeOut(sums), FadeIn(memory),
                          Transform(focus_w, w_boards[1].highlight(j, 0, 2, GOLD, outer=True)),
                          Transform(focus_y, answers.highlight(j, 0, 1, GOLD, outer=True)),
                          w_boards[1].column(0).animate.set_opacity(0.42),
                          w_boards[1].column(1).animate.set_opacity(1),
                          Transform(self.status, place(label("计算 Y 的第 2 列", 26, GOLD), 3.10, 3.48, 3.48)),
                          run_time=0.75)
            # 第一列先看符号地址；第二列用同一套动作直接代数值。
            shown_x, shown_w = (x_names, w_names) if j == 0 else (x, w)
            terms, labels, total = self.assemble(j, x_boards[j], w_boards[j], shown_x, shown_w, w_names, y_names)
            if j == 0:
                numbers = [scaled_term(w[k, j], x[:, k], k) for k in range(3)]
                self.change_hint("代入数字，同一个权重同时乘这一列的两个数",
                                 ReplacementTransform(x_boards[0], x_boards[1]),
                                 ReplacementTransform(w_boards[0], w_boards[1]),
                                 *(ReplacementTransform(old, new) for old, new in zip(terms, numbers)),
                                 duration=1.0)
                self.wait(1.15)
                terms = self.distribute(numbers, x, w[:, j])
                self.wait(1.35)

            # 第 k 个加数是 W[k,j] 乘 x 的整列，之后在同一行内相加。
            weighted = x * w[:, j]
            products = [column_vector(weighted[:, k], TERM_X[k], COLORS[k]) for k in range(3)]
            self.change_hint("先得到三个加权列向量，再把同一行的数相加",
                             LaggedStart(*(ReplacementTransform(old, new) for old, new in zip(terms, products)),
                                         lag_ratio=0.22), duration=1.0)
            self.wait(0.85)
            result = column_vector(weighted.sum(axis=1), ANSWER_X, GOLD)
            sums = VGroup(*(note(f"第 {r + 1} 行相加", row_sum_line(weighted[r], y_names[r, j]), xpos)
                            for r, xpos in enumerate((-6.52, 0.56))))
            self.play(FadeOut(memory), FadeIn(sums), ReplacementTransform(total, result), run_time=0.7)
            self.wait(1.15)
            self.play(*(FadeOut(targets.at(r, j)) for r in range(2)),
                      *(TransformFromCopy(result.at(r, 0), answers.at(r, j), path_arc=0.16) for r in range(2)),
                      run_time=1.05)
            self.wait(0.7)
            work = VGroup(labels, *products, result)

        self.play(FadeOut(work), FadeOut(sums), FadeOut(focus_w), FadeOut(focus_y),
                  w_boards[1].column(0).animate.set_opacity(1),
                  Transform(self.status, place(label("Y = xW", 28, GOLD), 3.10, 3.48, 3.48)), run_time=0.55)
        recap = self.annotations(0, w_names)
        example = [scaled_term(w[k, 0], x[:, k], k) for k in range(3)]
        self.change_hint("回看第一列，三个权重沿 W 的第一列从上往下取",
                         FadeIn(recap), FadeIn(VGroup(*example, column_vector(y[:, 0], ANSWER_X, GOLD))),
                         FadeIn(memory), duration=0.65)
        self.wait(3.2)

    @staticmethod
    def hint_text(text):
        return place(label(text, 26), -6.52, -0.02, 13.0)

    def change_hint(self, text, *animations, duration=0.43):
        following = self.hint_text(text)
        self.play(FadeOut(self.hint, shift=0.05 * UP), FadeIn(following, shift=0.05 * UP),
                  *animations, run_time=duration)
        self.hint = following

    def annotations(self, j, w_names):
        titles = [label(f"{w_names[k, j]} × x 的第 {k + 1} 列", 21, COLORS[k]).move_to([TERM_X[k], -0.69, 0])
                  for k in range(3)]
        signs = [label("+", 32).move_to([(TERM_X[k] + TERM_X[k + 1]) / 2, WORK_Y, 0]) for k in range(2)]
        equals = label("=", 32).move_to([3.52, WORK_Y, 0])
        destination = label(f"Y 的第 {j + 1} 列", 22, GOLD).move_to([ANSWER_X, -0.69, 0])
        return VGroup(VGroup(*titles), VGroup(*signs), equals, destination)

    def assemble(self, j, x_board, w_board, shown_x, shown_w, w_names, y_names):
        marks = self.annotations(j, w_names)
        titles, signs, equals, destination = marks
        terms = []
        for k in range(3):
            tint = COLORS[k]
            boxes = VGroup(x_board.highlight(k, 0, 1, tint), w_board.highlight(j, k, k, tint))
            term = scaled_term(shown_w[k, j], shown_x[:, k], k)
            self.change_hint(f"{w_names[k, j]} 在 W 的第 {k + 1} 行，乘 x 的第 {k + 1} 列",
                             FadeIn(boxes), FadeIn(titles[k]), duration=0.42 if j == 0 else 0.28)
            self.wait(0.38 if j == 0 else 0.20)
            signs_in = [FadeIn(signs[k - 1])] if k else []
            self.play(TransformFromCopy(w_board.at(k, j), term[0], path_arc=-0.20),
                      *(TransformFromCopy(x_board.at(r, k), term[2].at(r, 0), path_arc=0.12) for r in range(2)),
                      FadeIn(term[1]), FadeIn(term[2].brackets), *signs_in, run_time=1.23 if j == 0 else 0.80)
            self.add(term)
            self.wait(0.53 if j == 0 else 0.28)
            self.play(FadeOut(boxes), run_time=0.18)
            terms.append(term)
        total = column_vector(y_names[:, j], ANSWER_X, GOLD)
        self.play(FadeIn(equals), FadeIn(destination), FadeIn(total), run_time=0.43)
        self.add(marks)
        self.wait(0.85 if j == 0 else 0.45)
        return terms, marks, total

    def distribute(self, terms, x, weights):
        expanded, animations = [], []
        for k, term in enumerate(terms):
            color = COLORS[k]
            rows = []
            for value in x[:, k]:
                operand = scalar(value) if value >= 0 else f"({scalar(value)})"
                rows.append([VGroup(label(scalar(weights[k]), 28, color), label("×", 22, MUTED),
                                    label(operand, 28, color)).arrange(RIGHT, buff=0.10)])
            product = DisplayMatrix(rows, dx=2.35).move_to([TERM_X[k], WORK_Y, 0])
            transfers = [FadeOut(term), FadeIn(product.brackets)]
            for r in range(2):
                weight, times, value = product.at(r, 0)
                transfers.extend((TransformFromCopy(term[0], weight), FadeIn(times),
                                  TransformFromCopy(term[2].at(r, 0), value)))
            animations.append(AnimationGroup(*transfers))
            expanded.append(product)
        self.play(LaggedStart(*animations, lag_ratio=0.22), run_time=1.85)
        self.add(*expanded)
        return expanded
