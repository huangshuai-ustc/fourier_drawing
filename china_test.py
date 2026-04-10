import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from matplotlib.font_manager import FontProperties
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# ============ 中文字体 ============
CN_FONT = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')

# 国旗颜色
CHRED = '#EE1C25'
CHYEL = '#FFFF00'


# ================================================================
# 1. 根据 LaTeX 精确定义生成星星顶点
# ================================================================

def create_star_vertices_latex():
    """
    LaTeX 模板中的五角星10个顶点（外-内交替）
    模板中顶点最远距离原点 = 4
    """
    s5 = np.sqrt(5)
    vertices = np.array([
        [0, 4],
        [np.sqrt(50 - 22 * s5), -1 + s5],
        [np.sqrt(10 + 2 * s5), -1 + s5],
        [2 * np.sqrt(5 - 2 * s5), 4 - 2 * s5],
        [np.sqrt(10 - 2 * s5), -1 - s5],
        [0, -6 + 2 * s5],
        [-np.sqrt(10 - 2 * s5), -1 - s5],
        [-2 * np.sqrt(5 - 2 * s5), 4 - 2 * s5],
        [-np.sqrt(10 + 2 * s5), -1 + s5],
        [-np.sqrt(50 - 22 * s5), -1 + s5],
    ])
    return vertices


def transform_vertices(vertices, shift_x, shift_y, scale, rotate_deg):
    """
    模拟 LaTeX tikz 的 scope 变换链：缩放 → 旋转 → 平移
    然后从 LaTeX 坐标系（中心原点, y向上）转到画布坐标系（左上原点, y向下）
    """
    v = vertices.copy() * scale

    angle_rad = np.radians(rotate_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    v = v @ rot.T

    v[:, 0] += shift_x
    v[:, 1] += shift_y

    # LaTeX → 画布: x_new = x+15, y_new = 10-y
    v[:, 0] = v[:, 0] + 15
    v[:, 1] = 10 - v[:, 1]
    return v


# ================================================================
# 2. 沿闭合多边形边缘密集采样
# ================================================================

def sample_closed_polygon(vertices, num_points):
    """
    沿闭合多边形的边缘均匀采样，返回 num_points 个点
    保证首尾相接（不重复首点）
    """
    n = len(vertices)
    edges = []
    total_len = 0.0
    for i in range(n):
        seg = vertices[(i + 1) % n] - vertices[i]
        length = np.linalg.norm(seg)
        edges.append(length)
        total_len += length

    points = []
    for i in range(n):
        n_seg = max(int(round(num_points * edges[i] / total_len)), 1)
        start = vertices[i]
        end = vertices[(i + 1) % n]
        for j in range(n_seg):
            t = j / n_seg
            points.append(start + t * (end - start))

    points = np.array(points)
    if len(points) > num_points:
        indices = np.linspace(0, len(points) - 1, num_points, dtype=int)
        points = points[indices]
    elif len(points) < num_points:
        old_t = np.linspace(0, 1, len(points), endpoint=False)
        new_t = np.linspace(0, 1, num_points, endpoint=False)
        points = np.column_stack([
            np.interp(new_t, old_t, points[:, 0]),
            np.interp(new_t, old_t, points[:, 1]),
        ])

    return points


def sample_closed_rectangle(width, height, num_points):
    """采样矩形边缘"""
    vertices = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ])
    return sample_closed_polygon(vertices, num_points)


# ================================================================
# 3. 生成六条闭合轮廓
# ================================================================

def generate_all_contours(samples_per_contour=2000):
    """
    生成6条独立的闭合轮廓:
      0: 矩形边框 30×20
      1: 大五角星
      2-5: 四颗小五角星
    返回 list of (points_array, color_str, label_str)
    """
    outer_shift = (-10, 5)

    large_star = transform_vertices(
        create_star_vertices_latex(),
        shift_x=outer_shift[0], shift_y=outer_shift[1],
        scale=3 / 4, rotate_deg=0,
    )

    small_params = [
        (5, 3, 90 + np.degrees(np.arctan(3 / 5))),
        (7, 1, 90 + np.degrees(np.arctan(1 / 7))),
        (7, -2, 90 - np.degrees(np.arctan(2 / 7))),
        (5, -4, 90 - np.degrees(np.arctan(4 / 5))),
    ]
    small_stars = []
    for dx, dy, rot in small_params:
        star = transform_vertices(
            create_star_vertices_latex(),
            shift_x=outer_shift[0] + dx,
            shift_y=outer_shift[1] + dy,
            scale=1 / 4, rotate_deg=rot,
        )
        small_stars.append(star)

    contours = []

    rect_pts = sample_closed_rectangle(30, 20, samples_per_contour)
    contours.append((rect_pts, '#FFFFFF', '矩形边框'))

    large_pts = sample_closed_polygon(large_star, samples_per_contour)
    contours.append((large_pts, CHYEL, '大星'))

    for i, star in enumerate(small_stars):
        pts = sample_closed_polygon(star, samples_per_contour)
        contours.append((pts, CHYEL, f'小星{i + 1}'))

    return contours, [large_star] + small_stars


# ================================================================
# 4. 单条轮廓的傅里叶分析器
# ================================================================

class SingleContourFourier:
    """对一条闭合轮廓做傅里叶分析"""

    def __init__(self, points, max_harmonics=150, label=''):
        self.label = label
        self.points = np.asarray(points, dtype=float)
        self.max_harmonics = max_harmonics

        if not np.allclose(self.points[0], self.points[-1]):
            self.points = np.vstack([self.points, self.points[0]])

        self._compute()

    def _compute(self):
        n = len(self.points)
        z = self.points[:, 0] + 1j * self.points[:, 1]
        coeffs = fft(z) / n

        amplitudes = np.abs(coeffs)
        order = np.argsort(amplitudes)[::-1]

        self.epicycles = []
        for idx in order:
            amp = amplitudes[idx]
            if amp < 1e-9:
                continue
            freq = idx if idx <= n // 2 else idx - n
            phase = np.angle(coeffs[idx])
            self.epicycles.append({'freq': freq, 'amp': amp, 'phase': phase})
            if len(self.epicycles) >= self.max_harmonics:
                break

        self.epicycles.sort(key=lambda e: abs(e['freq']))

    def evaluate(self, t):
        """t ∈ [0,1] → (x, y)"""
        z = sum(
            e['amp'] * np.exp(1j * (2 * np.pi * e['freq'] * t + e['phase']))
            for e in self.epicycles
        )
        return z.real, z.imag

    def epicycle_chain(self, t):
        """返回圆环链各节点 [(x0,y0), (x1,y1), ...]"""
        cx, cy = 0.0, 0.0
        chain = [(cx, cy)]
        for e in self.epicycles:
            angle = 2 * np.pi * e['freq'] * t + e['phase']
            cx += e['amp'] * np.cos(angle)
            cy += e['amp'] * np.sin(angle)
            chain.append((cx, cy))
        return chain


# ================================================================
# 5. 静态国旗
# ================================================================

def draw_static_flag(star_vertices_list):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(-1, 31)
    ax.set_ylim(21, -1)
    ax.set_aspect('equal')
    ax.set_title('中华人民共和国国旗',
                 fontproperties=CN_FONT, fontsize=16, fontweight='bold')

    ax.add_patch(plt.Rectangle((0, 0), 30, 20,
                               facecolor=CHRED, edgecolor='black', linewidth=2))

    # names = ['大星', '小星1', '小星2', '小星3', '小星4']
    for i, verts in enumerate(star_vertices_list):
        ax.add_patch(Polygon(verts, closed=True,
                             facecolor=CHYEL, edgecolor='gold', linewidth=1))
        cx, cy = verts.mean(axis=0)
        # ax.text(cx, cy + (1.8 if i == 0 else 0.9), names[i],
        #         ha='center', fontproperties=CN_FONT,
        #         fontsize=8, color='white', fontweight='bold')

    # ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.savefig('chinese_flag_static.png', dpi=150, bbox_inches='tight')
    plt.show()


# ================================================================
# 6. 六路傅里叶同时绘制动画
# ================================================================

def create_multi_fourier_animation(contours, star_vertices_list,
                                   frames=3000, interval=16,
                                   max_harmonics=200,
                                   num_display_circles=25):
    """
    6 条轮廓各自独立做傅里叶，同时绘制
    左图: 6 组圆环机构
    右图: 6 条轨迹逐渐成形
    """

    # --- 构建傅里叶分析器 ---
    analyzers = []
    for pts, color, label in contours:
        fa = SingleContourFourier(pts, max_harmonics=max_harmonics, label=label)
        analyzers.append(fa)
        print(f"  [{label}] 使用 {len(fa.epicycles)} 个傅里叶分量")

    # --- 预计算范围 ---
    all_test = []
    for fa in analyzers:
        for t in np.linspace(0, 1, 500):
            all_test.append(fa.evaluate(t))
    all_test = np.array(all_test)
    margin = 2
    x_min, x_max = all_test[:, 0].min() - margin, all_test[:, 0].max() + margin
    y_min, y_max = all_test[:, 1].min() - margin, all_test[:, 1].max() + margin

    # --- 配色 ---
    trace_colors = ['#FFFFFF', '#FFD700',
                    '#FFA500', '#FF6347', '#FF69B4', '#DA70D6']

    # --- 创建画布 ---
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in [ax_left, ax_right]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # y 翻转
        ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.1, color='gray', linestyle='--')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_color('gray')
            spine.set_alpha(0.3)

    ax_left.set_title('6 组傅里叶圆环机构', fontproperties=CN_FONT,
                      fontsize=14, fontweight='bold', color='white')
    ax_right.set_title('六路同时绘制', fontproperties=CN_FONT,
                       fontsize=14, fontweight='bold', color='white')

    # 右图：淡色目标轮廓
    for pts, color, label in contours:
        ax_right.plot(pts[:, 0], pts[:, 1], '--',
                      color=color, alpha=0.15, linewidth=0.8)

    # --- 为每条轮廓创建绘图对象 ---
    circle_objs = []
    chain_lines = []
    pen_lefts = []
    pen_rights = []
    trace_lines = []
    trace_data = []

    for k, (fa, (_, color, label)) in enumerate(zip(analyzers, contours)):
        tc = trace_colors[k]

        n_show = min(num_display_circles, len(fa.epicycles))
        circles_k = []
        for i in range(n_show):
            amp = fa.epicycles[i]['amp']
            c = Circle((0, 0), amp, fill=False,
                       edgecolor=tc, linewidth=0.6, alpha=0.35)
            ax_left.add_patch(c)
            circles_k.append(c)
        circle_objs.append(circles_k)

        line, = ax_left.plot([], [], '-', color=tc, linewidth=0.7, alpha=0.5)
        chain_lines.append(line)

        pl, = ax_left.plot([], [], 'o', color=tc, markersize=4, zorder=10)
        pr, = ax_right.plot([], [], 'o', color=tc, markersize=4, zorder=10)
        pen_lefts.append(pl)
        pen_rights.append(pr)

        lw = 2.5 if k == 0 else (2.0 if k == 1 else 1.5)
        tl, = ax_right.plot([], [], '-', color=tc, linewidth=lw, alpha=0.9)
        trace_lines.append(tl)
        trace_data.append(([], []))

    # 进度文字
    info_text = fig.text(0.5, 0.02, '', ha='center',
                         fontproperties=CN_FONT, fontsize=12, color='white',
                         bbox=dict(boxstyle='round', facecolor='#2d2d44', alpha=0.9))

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = []
    for k, (_, color, label) in enumerate(contours):
        legend_elements.append(
            Line2D([0], [0], color=trace_colors[k], linewidth=2, label=label))
    ax_right.legend(handles=legend_elements, loc='upper right',
                    fontsize=9, facecolor='#2d2d44', edgecolor='gray',
                    labelcolor='white', prop=CN_FONT)

    def update(frame):
        t = frame / frames

        for k, fa in enumerate(analyzers):
            chain = fa.epicycle_chain(t)
            n_show = len(circle_objs[k])

            for i in range(n_show):
                if i < len(chain) - 1:
                    circle_objs[k][i].center = chain[i]

            cx = [c[0] for c in chain[:n_show + 1]]
            cy = [c[1] for c in chain[:n_show + 1]]
            chain_lines[k].set_data(cx, cy)

            px, py = fa.evaluate(t)
            pen_lefts[k].set_data([px], [py])
            pen_rights[k].set_data([px], [py])

            trace_data[k][0].append(px)
            trace_data[k][1].append(py)
            trace_lines[k].set_data(trace_data[k][0], trace_data[k][1])

        progress = t * 100
        info_text.set_text(
            f'绘制进度: {progress:.1f}%  |  '
            f'6 条独立傅里叶同时绘制  |  '
            f'每条 ≤{max_harmonics} 个谐波'
        )

        if progress >= 99.5:
            ax_right.set_title('✓ 六路傅里叶绘制完成！', fontproperties=CN_FONT,
                               fontsize=14, fontweight='bold', color='#00ff88')

        return (chain_lines + pen_lefts + pen_rights +
                trace_lines + [info_text] +
                [c for cl in circle_objs for c in cl])

    anim = FuncAnimation(fig, update, frames=frames,
                         interval=interval, blit=False, repeat=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07)
    return anim, fig


# ================================================================
# 7. 主程序
# ================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("六路傅里叶同时绘制五星红旗")
    print("每个封闭图形（矩形 + 5 颗星）各自独立傅里叶，无跳跃")
    print("=" * 70)

    # 生成轮廓
    print("\n1. 生成 6 条闭合轮廓...")
    contours, star_verts = generate_all_contours(samples_per_contour=3000)
    for pts, color, label in contours:
        print(f"   [{label}] {len(pts)} 个采样点  "
              f"X:[{pts[:, 0].min():.2f}, {pts[:, 0].max():.2f}]  "
              f"Y:[{pts[:, 1].min():.2f}, {pts[:, 1].max():.2f}]")

    # 静态国旗
    print("\n2. 显示静态国旗...")
    draw_static_flag(star_verts)

    # 动画
    print("\n3. 构建六路傅里叶动画...")
    anim, fig = create_multi_fourier_animation(
        contours, star_verts,
        frames=3000,
        interval=16,
        max_harmonics=200,
        num_display_circles=25,
    )

    # 保存
    print("\n4. 保存动画...")
    try:
        anim.save('flag_6fourier.gif', writer='pillow', fps=30, dpi=80)
        print("   ✓ 已保存 flag_6fourier.gif")
    except Exception as e:
        print(f"   gif 保存失败: {e}")
        try:
            anim.save('flag_6fourier.mp4', writer='ffmpeg', fps=30, dpi=100)
            print("   ✓ 已保存 flag_6fourier.mp4")
        except Exception as e2:
            print(f"   mp4 也失败: {e2}")

    print("\n5. 显示动画...")
    plt.show()
    print("\n✓ 完成！")