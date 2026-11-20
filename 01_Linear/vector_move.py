from manim import *


class VectorMove(Scene):
    def construct(self):

        # 1. 创建二维坐标系
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-1, 6, 1],
            x_length=10,
            y_length=7,
            axis_config={
                "include_numbers": True,
            },
            tips=False,
        )

        labels = axes.get_axis_labels(x_label="x", y_label="y")

        self.play(Create(axes), Write(labels))

        # 2. 定义三个关键位置
        origin = axes.c2p(0, 0)
        position_1 = axes.c2p(-3, 0)
        position_2 = axes.c2p(-3, 4)

        # 3. 创建初始点
        dot = Dot(origin, color=YELLOW)
        self.play(FadeIn(dot))

        # 4. 第一步：向左移动3个单位
        horizontal_line = Line(origin, position_1, color=BLUE, stroke_width=4)

        self.play(dot.animate.move_to(position_1), Create(horizontal_line), run_time=1.5)

        # 5. 第二步：向上移动4个单位
        vertical_line = Line(position_1, position_2, color=GREEN, stroke_width=4)

        self.play(dot.animate.move_to(position_2), Create(vertical_line), run_time=1.5)

        # 6. 显示最终坐标
        coordinate = MathTex("(-3,4)", color=YELLOW).next_to(dot, UR)

        self.play(Write(coordinate))

        # 7. 从原点绘制最终向量
        vector = Arrow(origin, position_2, buff=0.15, color=RED, stroke_width=6)

        self.play(GrowArrow(vector))

        self.wait(2)
