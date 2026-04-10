import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.font_manager import FontProperties
from matplotlib.collections import PolyCollection
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# ============ 中文字体 ============
# macOS
# CN_FONT = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
# Windows
# CN_FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
# Linux
# CN_FONT = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')

import platform
_sys = platform.system()
if _sys == 'Darwin':
    CN_FONT = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')
elif _sys == 'Windows':
    CN_FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
else:
    try:
        CN_FONT = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    except:
        CN_FONT = FontProperties()

# 国旗颜色
CHRED = '#EE1C25'
CHYEL = '#FFFF00'


# ================================================================
# 1. 根据 LaTeX 精确定义生成星星顶点
# ================================================================

def create_star_vertices_latex():
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
    v = vertices.copy() * scale
    angle_rad = np.radians(rotate_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    v = v @ rot.T
    v[:, 0] += shift_x
    v[:, 1] += shift_y
    v[:, 0] = v[:, 0] + 15
    v[:, 1] = 10 - v[:, 1]
    return v


# ================================================================
# 2. 沿闭合多边形边缘密集采样
# ================================================================

def sample_closed_polygon(vertices, num_points):
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
    vertices = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ])
    return sample_closed_polygon(vertices, num_points)


# ================================================================
# 3. 生成六条闭合轮廓
# ================================================================

def generate_all_contours(samples_per_contour=2000):
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
        z = sum(
            e['amp'] * np.exp(1j * (2 * np.pi * e['freq'] * t + e['phase']))
            for e in self.epicycles
        )
        return z.real, z.imag

    def evaluate_array(self, t_array):
        """向量化批量计算，返回 (N,2) 数组"""
        t_array = np.asarray(t_array)
        z = np.zeros(len(t_array), dtype=complex)
        for e in self.epicycles:
            angles = 2 * np.pi * e['freq'] * t_array + e['phase']
            z += e['amp'] * np.exp(1j * angles)
        return np.column_stack([z.real, z.imag])

    def epicycle_chain(self, t):
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

    for i, verts in enumerate(star_vertices_list):
        ax.add_patch(Polygon(verts, closed=True,
                             facecolor=CHYEL, edgecolor='gold', linewidth=1))

    plt.tight_layout()
    plt.savefig('chinese_flag_static.png', dpi=150, bbox_inches='tight')
    plt.show()


# ================================================================
# 6. 六路傅里叶同时绘制动画（带渐进填色）
# ================================================================

def create_multi_fourier_animation(contours, star_vertices_list,
                                   frames=3000, interval=16,
                                   max_harmonics=200,
                                   num_display_circles=25):
    """
    6 条轮廓各自独立做傅里叶，同时绘制。
    ★ 新增: 右图随画笔走过的轨迹逐步填充颜色
       - 矩形边框 → 填充红色（国旗底色）
       - 大星 / 小星 → 填充金黄色
       填充区域 = 当前已绘制的轨迹点构成的闭合多边形
       alpha 随完成度从 0.15 渐变到 1.0
    """

    # ---- 构建傅里叶分析器 ----
    analyzers = []
    for pts, color, label in contours:
        fa = SingleContourFourier(pts, max_harmonics=max_harmonics, label=label)
        analyzers.append(fa)
        print(f"  [{label}] {len(fa.epicycles)} 个傅里叶分量")

    # ---- 预计算完整轨迹（用于确定坐标范围 & 最终填充） ----
    full_traces = []
    for fa in analyzers:
        pts = fa.evaluate_array(np.linspace(0, 1, frames, endpoint=False))
        full_traces.append(pts)

    all_pts = np.vstack(full_traces)
    margin = 2
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin

    # ---- 每条轮廓的填充颜色 ----
    #   0: 矩形 → 红色底
    #   1: 大星 → 金黄
    #   2-5: 小星 → 金黄
    fill_colors = [CHRED, CHYEL, CHYEL, CHYEL, CHYEL, CHYEL]

    # 轨迹线颜色（亮色，用于画笔路径）
    trace_colors = ['#EE1C25', '#FFFF00',
                    '#FFFF00', '#FFFF00', '#FFFF00', '#FFFF00']

    # ---- 创建画布 ----
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor("#000000")

    for ax in [ax_left, ax_right]:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.set_aspect('equal')
        ax.set_facecolor('#ffffff')
        ax.grid(True, alpha=0.1, color='gray', linestyle='--')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_color('gray')
            spine.set_alpha(0.3)

    ax_left.set_title('6 组傅里叶圆环机构', fontproperties=CN_FONT,
                      fontsize=14, fontweight='bold', color='white')
    ax_right.set_title('六路同时绘制（渐进填色）', fontproperties=CN_FONT,
                       fontsize=14, fontweight='bold', color='white')

    # 右图：极淡目标轮廓
    for pts, color, label in contours:
        ax_right.plot(pts[:, 0], pts[:, 1], '--',
                      color=color, alpha=0.08, linewidth=0.5)

    # ---- 为每条轮廓创建绘图对象 ----
    circle_objs = []
    chain_lines = []
    pen_lefts = []
    pen_rights = []
    trace_lines = []
    trace_data = []          # (list_x, list_y) 每帧追加
    fill_patches = []        # 右图的填充 Polygon

    for k, (fa, (_, color, label)) in enumerate(zip(analyzers, contours)):
        tc = trace_colors[k]

        # 圆环（左图）
        n_show = min(num_display_circles, len(fa.epicycles))
        circles_k = []
        for i in range(n_show):
            amp = fa.epicycles[i]['amp']
            c = Circle((0, 0), amp, fill=False,
                       edgecolor=tc, linewidth=0.6, alpha=0.35)
            ax_left.add_patch(c)
            circles_k.append(c)
        circle_objs.append(circles_k)

        # 连线
        line, = ax_left.plot([], [], '-', color=tc, linewidth=0.7, alpha=0.5)
        chain_lines.append(line)

        # 笔尖
        pl, = ax_left.plot([], [], 'o', color=tc, markersize=4, zorder=10)
        pr, = ax_right.plot([], [], 'o', color=tc, markersize=4, zorder=15)
        pen_lefts.append(pl)
        pen_rights.append(pr)

        # 轨迹线（右图，画在填充之上）
        lw = 2.0 if k == 0 else (1.8 if k == 1 else 1.3)
        tl, = ax_right.plot([], [], '-', color=tc, linewidth=lw, alpha=0.85,
                            zorder=12)
        trace_lines.append(tl)
        trace_data.append(([], []))

        # ★ 填充多边形（右图，zorder 低于轨迹线）
        #    初始为不可见的空多边形
        fill_poly = Polygon([[0, 0]], closed=True,
                            facecolor=fill_colors[k],
                            edgecolor='none',
                            alpha=0.0,       # 初始透明
                            zorder=5)
        ax_right.add_patch(fill_poly)
        fill_patches.append(fill_poly)

    # 进度文字
    info_text = fig.text(0.5, 0.02, '', ha='center',
                         fontproperties=CN_FONT, fontsize=12, color='white',
                         bbox=dict(boxstyle='round', facecolor='#2d2d44',
                                   alpha=0.9))

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = []
    for k, (_, color, label) in enumerate(contours):
        legend_elements.append(
            Line2D([0], [0], color=trace_colors[k], linewidth=2, label=label))
    ax_right.legend(handles=legend_elements, loc='upper right',
                    fontsize=9, facecolor='#2d2d44', edgecolor='gray',
                    labelcolor='white', prop=CN_FONT)

    # ---- 填色参数 ----
    # 在绘制前 fill_start_frac 不填色，之后 alpha 线性增长
    fill_start_frac = 0.10   # 画了 10% 之后才开始出现填色
    fill_full_frac = 0.95    # 画到 95% 时填色完全不透明

    # 为了性能，填充多边形不是每帧都用全部轨迹点
    # 而是每隔 thin_every 帧取一个点来构建多边形
    thin_every = max(frames // 600, 1)

    def update(frame):
        t = frame / frames
        progress = t * 100

        for k, fa in enumerate(analyzers):
            # --- 圆环链 ---
            chain = fa.epicycle_chain(t)
            n_show = len(circle_objs[k])
            for i in range(n_show):
                if i < len(chain) - 1:
                    circle_objs[k][i].center = chain[i]

            cx_chain = [c[0] for c in chain[:n_show + 1]]
            cy_chain = [c[1] for c in chain[:n_show + 1]]
            chain_lines[k].set_data(cx_chain, cy_chain)

            # --- 笔尖 ---
            px, py = fa.evaluate(t)
            pen_lefts[k].set_data([px], [py])
            pen_rights[k].set_data([px], [py])

            # --- 轨迹 ---
            trace_data[k][0].append(px)
            trace_data[k][1].append(py)
            trace_lines[k].set_data(trace_data[k][0], trace_data[k][1])

            # --- ★ 渐进填色 ---
            n_pts = len(trace_data[k][0])
            if t > fill_start_frac and n_pts >= 3:
                # 计算当前 alpha
                if t >= fill_full_frac:
                    alpha = 1.0
                else:
                    alpha = (t - fill_start_frac) / (fill_full_frac - fill_start_frac)
                    alpha = np.clip(alpha, 0.0, 1.0)
                    # 用 ease-in 曲线让填色更自然
                    alpha = alpha ** 1.5

                # 稀疏取点构建多边形（性能优化）
                tx = trace_data[k][0]
                ty = trace_data[k][1]
                step = max(1, n_pts // 400)  # 最多 ~400 个顶点
                idx = list(range(0, n_pts, step))
                if idx[-1] != n_pts - 1:
                    idx.append(n_pts - 1)

                poly_verts = np.column_stack([
                    [tx[i] for i in idx],
                    [ty[i] for i in idx],
                ])

                fill_patches[k].set_xy(poly_verts)
                fill_patches[k].set_alpha(alpha)
            else:
                fill_patches[k].set_alpha(0.0)

        info_text.set_text(
            f'绘制进度: {progress:.1f}%  |  '
            f'6 条独立傅里叶 · 渐进填色  |  '
            f'每条 ≤{max_harmonics} 个谐波'
        )

        if progress >= 99.5:
            ax_right.set_title('✓ 绘制完成！', fontproperties=CN_FONT,
                               fontsize=14, fontweight='bold', color='#00ff88')
            # 最终帧：用完整预计算轨迹做精确填充
            for k in range(len(analyzers)):
                fill_patches[k].set_xy(full_traces[k])
                fill_patches[k].set_alpha(1.0)

        return (chain_lines + pen_lefts + pen_rights +
                trace_lines + fill_patches + [info_text] +
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
    print("六路傅里叶同时绘制五星红旗 —— 渐进填色版")
    print("=" * 70)

    print("\n1. 生成 6 条闭合轮廓...")
    contours, star_verts = generate_all_contours(samples_per_contour=3000)
    for pts, color, label in contours:
        print(f"   [{label}] {len(pts)} 点  "
              f"X:[{pts[:, 0].min():.2f},{pts[:, 0].max():.2f}]  "
              f"Y:[{pts[:, 1].min():.2f},{pts[:, 1].max():.2f}]")

    print("\n2. 静态国旗...")
    draw_static_flag(star_verts)

    print("\n3. 构建动画...")
    anim, fig = create_multi_fourier_animation(
        contours, star_verts,
        frames=3000,
        interval=16,
        max_harmonics=200,
        num_display_circles=25,
    )

    print("\n4. 保存...")
    try:
        anim.save('flag_6fourier_fill.gif', writer='pillow', fps=30, dpi=80)
        print("   ✓ flag_6fourier_fill.gif")
    except Exception as e:
        print(f"   gif 失败: {e}")
        try:
            anim.save('flag_6fourier_fill.mp4', writer='ffmpeg', fps=30, dpi=100)
            print("   ✓ flag_6fourier_fill.mp4")
        except Exception as e2:
            print(f"   mp4 也失败: {e2}")

    print("\n5. 显示...")
    plt.show()
    print("\n✓ 完成！")