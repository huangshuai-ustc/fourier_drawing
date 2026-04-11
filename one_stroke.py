import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.font_manager import FontProperties
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# ============ 中文字体 ============
CN_FONT = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')

# 国旗颜色
CHRED = '#EE1C25'
CHYEL = '#FFFF00'


def create_star_vertices_latex(scale=1.0):
    s5 = np.sqrt(5)
    vertices = np.array([
        [0, 4],
        [np.sqrt(50 - 22*s5), -1 + s5],
        [np.sqrt(10 + 2*s5), -1 + s5],
        [2*np.sqrt(5 - 2*s5), 4 - 2*s5],
        [np.sqrt(10 - 2*s5), -1 - s5],
        [0, -6 + 2*s5],
        [-np.sqrt(10 - 2*s5), -1 - s5],
        [-2*np.sqrt(5 - 2*s5), 4 - 2*s5],
        [-np.sqrt(10 + 2*s5), -1 + s5],
        [-np.sqrt(50 - 22*s5), -1 + s5],
    ])
    return vertices * scale


def transform_star_to_flag(vertices, shift_x, shift_y, scale, rotate_deg):
    v = vertices * scale
    angle_rad = np.radians(rotate_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    v = v @ rot_matrix.T
    v[:, 0] += shift_x
    v[:, 1] += shift_y
    v[:, 0] = v[:, 0] + 15
    v[:, 1] = 10 - v[:, 1]
    return v


def interpolate_polygon_edges(vertices, num_points_per_edge):
    points = []
    n = len(vertices)
    for i in range(n):
        start = vertices[i]
        end = vertices[(i + 1) % n]
        for j in range(num_points_per_edge):
            t = j / num_points_per_edge
            points.append(start + t * (end - start))
    return np.array(points)


def get_flag_outline(num_samples=6000):
    outer_shift = (-10, 5)

    large_star_base = create_star_vertices_latex()
    large_star = transform_star_to_flag(
        large_star_base,
        shift_x=outer_shift[0] + 0,
        shift_y=outer_shift[1] + 0,
        scale=3/4,
        rotate_deg=0
    )

    small_star1 = transform_star_to_flag(
        create_star_vertices_latex(),
        shift_x=outer_shift[0] + 5,
        shift_y=outer_shift[1] + 3,
        scale=1/4,
        rotate_deg=90 + np.degrees(np.arctan(3/5))
    )

    small_star2 = transform_star_to_flag(
        create_star_vertices_latex(),
        shift_x=outer_shift[0] + 7,
        shift_y=outer_shift[1] + 1,
        scale=1/4,
        rotate_deg=90 + np.degrees(np.arctan(1/7))
    )

    small_star3 = transform_star_to_flag(
        create_star_vertices_latex(),
        shift_x=outer_shift[0] + 7,
        shift_y=outer_shift[1] + (-2),
        scale=1/4,
        rotate_deg=90 - np.degrees(np.arctan(2/7))
    )

    small_star4 = transform_star_to_flag(
        create_star_vertices_latex(),
        shift_x=outer_shift[0] + 5,
        shift_y=outer_shift[1] + (-4),
        scale=1/4,
        rotate_deg=90 - np.degrees(np.arctan(4/5))
    )

    all_stars = [large_star, small_star1, small_star2, small_star3, small_star4]

    width, height = 30, 20

    rect_vertices = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ])

    n_rect = num_samples // 3
    n_large = num_samples // 4
    n_small_each = num_samples // 12
    n_transitions = num_samples - n_rect - n_large - 4 * n_small_each
    n_trans_each = max(n_transitions // 6, 10)

    all_points = []

    rect_pts = interpolate_polygon_edges(rect_vertices, n_rect // 4)
    all_points.append(rect_pts)

    rect_end = rect_pts[-1]
    large_start = large_star[0]
    trans1 = np.linspace(rect_end, large_start, n_trans_each, endpoint=False)
    all_points.append(trans1)

    large_pts = interpolate_polygon_edges(large_star, n_large // 10)
    all_points.append(large_pts)

    prev_end = large_pts[-1]
    for i, star in enumerate([small_star1, small_star2, small_star3, small_star4]):
        trans = np.linspace(prev_end, star[0], n_trans_each, endpoint=False)
        all_points.append(trans)

        star_pts = interpolate_polygon_edges(star, n_small_each // 10)
        all_points.append(star_pts)
        prev_end = star_pts[-1]

    trans_back = np.linspace(prev_end, rect_pts[0], n_trans_each, endpoint=False)
    all_points.append(trans_back)

    points = np.vstack(all_points)

    return points, all_stars


def draw_static_flag(all_stars):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xlim(-1, 31)
    ax.set_ylim(21, -1)
    ax.set_aspect('equal')
    ax.set_title('中华人民共和国国旗',
                 fontproperties=CN_FONT, fontsize=16, fontweight='bold')
    ax.set_xlabel('X', fontproperties=CN_FONT)
    ax.set_ylabel('Y', fontproperties=CN_FONT)

    flag_bg = plt.Rectangle((0, 0), 30, 20,
                            facecolor=CHRED, edgecolor='black', linewidth=2)
    ax.add_patch(flag_bg)

    star_names = ['大星', '小星1', '小星2', '小星3', '小星4']
    for i, star in enumerate(all_stars):
        star_poly = Polygon(star, facecolor=CHYEL, edgecolor='gold', linewidth=1, closed=True)
        ax.add_patch(star_poly)
        cx, cy = star.mean(axis=0)

    plt.tight_layout()
    plt.savefig('chinese_national_flag.png', dpi=150, bbox_inches='tight')
    plt.show()
    return fig


class FourierDrawingAnimation:

    def __init__(self, points, max_harmonics=150):
        self.points = np.asarray(points, dtype=float)
        self.max_harmonics = max_harmonics

        if not np.allclose(self.points[0], self.points[-1]):
            self.points = np.vstack([self.points, self.points[0]])

        self.compute_fourier_coefficients()
        self.prepare_epicycles()

    def compute_fourier_coefficients(self):
        n_points = len(self.points)
        z = self.points[:, 0] + 1j * self.points[:, 1]
        self.coeffs = fft(z) / n_points
        self.n_points = n_points

    def prepare_epicycles(self):
        n = self.n_points
        amplitudes = np.abs(self.coeffs)

        indices = np.argsort(amplitudes)[::-1]

        self.epicycles = []
        count = 0
        for idx in indices:
            amp = amplitudes[idx]
            if amp < 1e-8:
                continue

            if idx <= n // 2:
                freq = idx
            else:
                freq = idx - n

            phase = np.angle(self.coeffs[idx])

            self.epicycles.append({
                'freq': freq,
                'amp': amp,
                'phase': phase,
            })
            count += 1
            if count >= self.max_harmonics * 2 + 1:
                break

        self.epicycles.sort(key=lambda x: abs(x['freq']))

        print(f"使用 {len(self.epicycles)} 个傅里叶分量")
        print(f"最大幅度: {max(e['amp'] for e in self.epicycles):.3f}")

    def evaluate(self, t):
        z = 0 + 0j
        for ep in self.epicycles:
            angle = 2 * np.pi * ep['freq'] * t + ep['phase']
            z += ep['amp'] * np.exp(1j * angle)
        return z.real, z.imag

    def evaluate_epicycle_chain(self, t):
        cx, cy = 0.0, 0.0
        chain = [(cx, cy)]

        for ep in self.epicycles:
            angle = 2 * np.pi * ep['freq'] * t + ep['phase']
            dx = ep['amp'] * np.cos(angle)
            dy = ep['amp'] * np.sin(angle)
            cx += dx
            cy += dy
            chain.append((cx, cy))

        return chain

    def create_animation(self, frames=3000, interval=20,
                        num_display_circles=40):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 9))

        # 预计算一圈轨迹来确定范围
        test_t = np.linspace(0, 1, 1000)
        test_pts = np.array([self.evaluate(t) for t in test_t])

        margin = 3
        x_min, x_max = test_pts[:, 0].min() - margin, test_pts[:, 0].max() + margin
        y_min, y_max = test_pts[:, 1].min() - margin, test_pts[:, 1].max() + margin

        for ax in [ax_left, ax_right]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, linestyle='--')

        ax_left.set_title('傅里叶圆环机构', fontproperties=CN_FONT,
                          fontsize=14, fontweight='bold')
        ax_right.set_title('绘制轨迹', fontproperties=CN_FONT,
                           fontsize=14, fontweight='bold')

        ax_left.plot(self.points[:, 0], self.points[:, 1],
                    'g--', alpha=0.2, linewidth=0.5)

        ax_right.plot(self.points[:, 0], self.points[:, 1],
                     'g--', alpha=0.3, linewidth=1)

        display_eps = self.epicycles[:min(num_display_circles, len(self.epicycles))]
        colors = plt.cm.hsv(np.linspace(0, 0.9, len(display_eps)))

        circles = []
        for i, ep in enumerate(display_eps):
            circle = Circle((0, 0), ep['amp'],
                          fill=False, edgecolor=colors[i],
                          linewidth=0.8, alpha=0.6)
            ax_left.add_patch(circle)
            circles.append(circle)

        chain_line, = ax_left.plot([], [], 'b-', linewidth=0.8, alpha=0.5)

        pen_left, = ax_left.plot([], [], 'ro', markersize=6, zorder=10)
        pen_right, = ax_right.plot([], [], 'ro', markersize=6, zorder=10)

        trace_line, = ax_right.plot([], [], 'r-', linewidth=1.8, alpha=0.9)

        info_text = fig.text(0.5, 0.02, '', ha='center', fontproperties=CN_FONT,
                            fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # ====== 预计算全部轨迹点 ======
        all_trace = np.array([self.evaluate(f / frames) for f in range(frames)])

        def update(frame):
            t = frame / frames

            chain = self.evaluate_epicycle_chain(t)

            for i, (ep, circle) in enumerate(zip(display_eps, circles)):
                if i < len(chain) - 1:
                    circle.center = chain[i]

            chain_x = [c[0] for c in chain]
            chain_y = [c[1] for c in chain]
            chain_line.set_data(chain_x, chain_y)

            final_x, final_y = chain[-1]
            pen_left.set_data([final_x], [final_y])
            pen_right.set_data([final_x], [final_y])

            # 用预计算的切片，不再 append
            seg = all_trace[:frame+1]
            trace_line.set_data(seg[:, 0], seg[:, 1])

            progress = t * 100
            info_text.set_text(f'绘制进度: {progress:.1f}%  |  '
                              f'傅里叶分量: {len(self.epicycles)}  |  '
                              f'已绘制点: {frame+1}')

            if progress >= 99.5:
                ax_right.set_title('✓ 绘制完成！', fontproperties=CN_FONT,
                                  fontsize=14, fontweight='bold', color='green')

            return [chain_line, pen_left, pen_right, trace_line, info_text] + circles

        anim = FuncAnimation(fig, update, frames=frames,
                           interval=interval, blit=False, repeat=False)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        return anim, fig


# ============ 主程序 ============
if __name__ == "__main__":
    print("=" * 70)
    print("傅里叶级数绘制五星红旗 - 基于 LaTeX 精确坐标")
    print("=" * 70)

    # 1. 生成轮廓点
    print("\n1. 生成国旗轮廓点...")
    points, all_stars = get_flag_outline(num_samples=8000)
    print(f"   共生成 {len(points)} 个轮廓点")
    print(f"   X范围: [{np.min(points[:, 0]):.2f}, {np.max(points[:, 0]):.2f}]")
    print(f"   Y范围: [{np.min(points[:, 1]):.2f}, {np.max(points[:, 1]):.2f}]")

    # 2. 显示静态国旗
    print("\n2. 显示静态国旗...")
    draw_static_flag(all_stars)

    # 3. 创建动画
    print("\n3. 创建傅里叶绘制动画...")
    drawer = FourierDrawingAnimation(points, max_harmonics=200)

    # ====== 关键参数：减少帧数 + 降低 dpi + 每帧间隔更短 ======
    FRAMES = 600          # 从 3000 降到 600
    FPS = 30
    INTERVAL = 1000 // FPS  # ~33ms

    anim, fig = drawer.create_animation(
        frames=FRAMES,
        interval=INTERVAL,
        num_display_circles=50
    )

    # 4. 保存动画
    print("\n4. 保存动画...")
    SAVE_DPI = 60  # 从 80 降到 60
    try:
        anim.save('one_stroke.gif',
                  writer='pillow', fps=FPS, dpi=SAVE_DPI)
        import os
        sz = os.path.getsize('one_stroke.gif') / (1024*1024)
        print(f"   ✓ 已保存 one_stroke.gif  ({sz:.1f} MB)")
    except Exception as e:
        print(f"   保存失败: {e}")
        print("   尝试保存为 mp4...")
        try:
            anim.save('one_stroke.mp4',
                      writer='ffmpeg', fps=FPS, dpi=SAVE_DPI)
            print("   ✓ 动画已保存为 one_stroke.mp4")
        except Exception as e2:
            print(f"   mp4 也保存失败: {e2}")

    # 5. 显示
    print("\n5. 显示动画窗口...")
    plt.show()

    print("\n✓ 完成！")