import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# 国旗颜色
CHRED = '#EE1C25'
CHYEL = '#FFFF00'


def create_star_vertices_latex(scale=1.0):
    """
    根据 LaTeX 模板创建五角星的10个顶点坐标
    LaTeX 中的星星模板（按顺序连接形成五角星）：
    (0,4), (sqrt(50-22*sqrt(5)), -1+sqrt(5)), (sqrt(10+2*sqrt(5)), -1+sqrt(5)),
    (2*sqrt(5-2*sqrt(5)), 4-2*sqrt(5)), (sqrt(10-2*sqrt(5)), -1-sqrt(5)),
    (0, -6+2*sqrt(5)), (-sqrt(10-2*sqrt(5)), -1-sqrt(5)),
    (-2*sqrt(5-2*sqrt(5)), 4-2*sqrt(5)), (-sqrt(10+2*sqrt(5)), -1+sqrt(5)),
    (-sqrt(50-22*sqrt(5)), -1+sqrt(5))
    """
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
    """
    对星星顶点进行变换（模拟 LaTeX 的 scope 变换）
    1. 先缩放
    2. 再旋转
    3. 再平移
    最后从 LaTeX 坐标系转换到左上角原点坐标系
    
    LaTeX 坐标系: 中心原点, y向上, 范围 (-15,-10)~(15,10)
    目标坐标系: 左上角原点, y向下, 范围 (0,0)~(30,20)
    """
    # 缩放
    v = vertices * scale
    
    # 旋转（LaTeX 中逆时针为正）
    angle_rad = np.radians(rotate_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    v = v @ rot_matrix.T
    
    # 平移（在 LaTeX 坐标系中）
    v[:, 0] += shift_x
    v[:, 1] += shift_y
    
    # 转换到左上角原点坐标系
    v[:, 0] = v[:, 0] + 15      # x_new = x_latex + 15
    v[:, 1] = 10 - v[:, 1]      # y_new = 10 - y_latex (翻转y轴)
    
    return v


def interpolate_polygon_edges(vertices, num_points_per_edge):
    """沿多边形的边进行线性插值，生成连续的轮廓点"""
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
    """
    获取五星红旗轮廓点，形成一条连续的闭合曲线
    使用与 LaTeX 完全一致的坐标和变换
    """
    # ============ 根据 LaTeX 代码计算各星星的位置 ============
    # 外层 scope: shift={(-10,5)}, scale=1, rotate=0
    outer_shift = (-10, 5)
    
    # 大星: shift={(0,0)}, scale=3/4, rotate=0
    # LaTeX 坐标中的位置: (-10+0, 5+0) = (-10, 5)
    large_star_base = create_star_vertices_latex()
    large_star = transform_star_to_flag(
        large_star_base,
        shift_x=outer_shift[0] + 0,
        shift_y=outer_shift[1] + 0,
        scale=3/4,
        rotate_deg=0
    )
    
    # 小星1: shift={(5,3)}, scale=1/4, rotate=90+atan(3/5)
    small_star1 = transform_star_to_flag(
        create_star_vertices_latex(),
        shift_x=outer_shift[0] + 5,
        shift_y=outer_shift[1] + 3,
        scale=1/4,
        rotate_deg=90 + np.degrees(np.arctan(3/5))
    )
    
    # 小星2: shift={(7,1)}, scale=1/4, rotate=90+atan(1/7)
    small_star2 = transform_star_to_flag(
        create_star_vertices_latex(),
        shift_x=outer_shift[0] + 7,
        shift_y=outer_shift[1] + 1,
        scale=1/4,
        rotate_deg=90 + np.degrees(np.arctan(1/7))
    )
    
    # 小星3: shift={(7,-2)}, scale=1/4, rotate=90-atan(2/7)
    small_star3 = transform_star_to_flag(
        create_star_vertices_latex(),
        shift_x=outer_shift[0] + 7,
        shift_y=outer_shift[1] + (-2),
        scale=1/4,
        rotate_deg=90 - np.degrees(np.arctan(2/7))
    )
    
    # 小星4: shift={(5,-4)}, scale=1/4, rotate=90-atan(4/5)
    small_star4 = transform_star_to_flag(
        create_star_vertices_latex(),
        shift_x=outer_shift[0] + 5,
        shift_y=outer_shift[1] + (-4),
        scale=1/4,
        rotate_deg=90 - np.degrees(np.arctan(4/5))
    )
    
    all_stars = [large_star, small_star1, small_star2, small_star3, small_star4]
    
    # ============ 构建一条连续的闭合曲线 ============
    # 策略：矩形 -> 跳到大星 -> 大星 -> 跳到小星1 -> ... -> 跳回矩形起点
    # 为了让傅里叶变换效果好，我们需要让路径尽量连续
    # 这里采用的方法：沿着每个形状的边缘采样，形状之间用直线连接
    
    width, height = 30, 20
    
    # 矩形的4个角（左上角原点坐标系）
    rect_vertices = np.array([
        [0, 0],       # 左上
        [width, 0],   # 右上
        [width, height],  # 右下
        [0, height],  # 左下
    ])
    
    # 分配采样点数
    n_rect = num_samples // 3          # 矩形用1/3的点
    n_large = num_samples // 4         # 大星用1/4的点
    n_small_each = num_samples // 12   # 每个小星
    n_transitions = num_samples - n_rect - n_large - 4 * n_small_each  # 过渡段
    n_trans_each = max(n_transitions // 6, 10)  # 6段过渡
    
    all_points = []
    
    # 1. 矩形边缘
    rect_pts = interpolate_polygon_edges(rect_vertices, n_rect // 4)
    all_points.append(rect_pts)
    
    # 2. 过渡：从矩形终点（左下角附近）到大星起点
    rect_end = rect_pts[-1]
    large_start = large_star[0]
    trans1 = np.linspace(rect_end, large_start, n_trans_each, endpoint=False)
    all_points.append(trans1)
    
    # 3. 大星边缘
    large_pts = interpolate_polygon_edges(large_star, n_large // 10)
    all_points.append(large_pts)
    
    # 4. 依次连接到各小星
    prev_end = large_pts[-1]
    for i, star in enumerate([small_star1, small_star2, small_star3, small_star4]):
        # 过渡到小星
        trans = np.linspace(prev_end, star[0], n_trans_each, endpoint=False)
        all_points.append(trans)
        
        # 小星边缘
        star_pts = interpolate_polygon_edges(star, n_small_each // 10)
        all_points.append(star_pts)
        prev_end = star_pts[-1]
    
    # 5. 过渡：从最后一个小星回到矩形起点
    trans_back = np.linspace(prev_end, rect_pts[0], n_trans_each, endpoint=False)
    all_points.append(trans_back)
    
    points = np.vstack(all_points)
    
    return points, all_stars


def draw_static_flag(all_stars):
    """绘制静态国旗"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.set_xlim(-1, 31)
    ax.set_ylim(21, -1)  # y轴反向
    ax.set_aspect('equal')
    ax.set_title('中华人民共和国国旗（根据 LaTeX 坐标精确绘制）', fontsize=16, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # 红色背景
    flag_bg = plt.Rectangle((0, 0), 30, 20,
                            facecolor=CHRED, edgecolor='black', linewidth=2)
    ax.add_patch(flag_bg)
    
    # 绘制五颗星
    star_names = ['大星', '小星1', '小星2', '小星3', '小星4']
    for i, star in enumerate(all_stars):
        star_poly = Polygon(star, facecolor=CHYEL, edgecolor='gold', linewidth=1, closed=True)
        ax.add_patch(star_poly)
        # 标注中心
        cx, cy = star.mean(axis=0)
        ax.plot(cx, cy, 'r+', markersize=8, markeredgewidth=1.5)
        ax.text(cx, cy + (1.5 if i == 0 else 0.8), star_names[i],
               ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
    
    # 打印星星中心坐标
    print("\n星星中心坐标（左上角原点坐标系）：")
    for i, star in enumerate(all_stars):
        cx, cy = star.mean(axis=0)
        print(f"  {star_names[i]}: ({cx:.2f}, {cy:.2f})")
    
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('chinese_flag_correct.png', dpi=150, bbox_inches='tight')
    plt.show()
    return fig


class FourierDrawingAnimation:
    """傅里叶级数绘制动画"""
    
    def __init__(self, points, max_harmonics=150):
        self.points = np.asarray(points, dtype=float)
        self.max_harmonics = max_harmonics
        
        # 确保闭合
        if not np.allclose(self.points[0], self.points[-1]):
            self.points = np.vstack([self.points, self.points[0]])
        
        self.compute_fourier_coefficients()
        self.prepare_epicycles()
        
    def compute_fourier_coefficients(self):
        """计算傅里叶系数"""
        n_points = len(self.points)
        z = self.points[:, 0] + 1j * self.points[:, 1]
        self.coeffs = fft(z) / n_points
        self.n_points = n_points
        
    def prepare_epicycles(self):
        """准备圆环数据，按幅度排序"""
        n = self.n_points
        amplitudes = np.abs(self.coeffs)
        
        # 选择幅度最大的谐波
        indices = np.argsort(amplitudes)[::-1]
        
        self.epicycles = []
        count = 0
        for idx in indices:
            amp = amplitudes[idx]
            if amp < 1e-8:
                continue
            
            # 频率：对应 fft 的频率
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
        
        # 按频率排序（从负到正）
        self.epicycles.sort(key=lambda x: abs(x['freq']))
        
        print(f"使用 {len(self.epicycles)} 个傅里叶分量")
        print(f"最大幅度: {max(e['amp'] for e in self.epicycles):.3f}")
        
    def evaluate(self, t):
        """
        在参数 t (0~1) 处计算傅里叶级数的值
        t=0 和 t=1 对应同一点（闭合曲线）
        """
        z = 0 + 0j
        for ep in self.epicycles:
            angle = 2 * np.pi * ep['freq'] * t + ep['phase']
            z += ep['amp'] * np.exp(1j * angle)
        return z.real, z.imag
    
    def evaluate_epicycle_chain(self, t):
        """
        计算参数 t 处每个圆环的中心位置（用于动画显示）
        返回所有圆环中心的列表
        """
        centers = [(0, 0)]  # 虚拟起点
        # 实际上第一个圆环的中心就是 DC 分量的位置
        # 我们从原点开始累加
        
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
        """创建绘制动画"""
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 9))
        
        # 计算坐标范围
        # 先预计算一圈的轨迹来确定范围
        test_t = np.linspace(0, 1, 1000)
        test_pts = np.array([self.evaluate(t) for t in test_t])
        
        margin = 3
        x_min, x_max = test_pts[:, 0].min() - margin, test_pts[:, 0].max() + margin
        y_min, y_max = test_pts[:, 1].min() - margin, test_pts[:, 1].max() + margin
        
        for ax in [ax_left, ax_right]:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)  # y轴反向（左上角原点）
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.2, linestyle='--')
        
        ax_left.set_title('傅里叶圆环机构', fontsize=14, fontweight='bold')
        ax_right.set_title('绘制轨迹', fontsize=14, fontweight='bold')
        
        # 左图：目标轮廓（淡色）
        ax_left.plot(self.points[:, 0], self.points[:, 1],
                    'g--', alpha=0.2, linewidth=0.5)
        
        # 右图：目标轮廓（淡色）
        ax_right.plot(self.points[:, 0], self.points[:, 1],
                     'g--', alpha=0.3, linewidth=1)
        
        # 创建圆环对象（左图）
        display_eps = self.epicycles[:min(num_display_circles, len(self.epicycles))]
        colors = plt.cm.hsv(np.linspace(0, 0.9, len(display_eps)))
        
        circles = []
        for i, ep in enumerate(display_eps):
            circle = Circle((0, 0), ep['amp'],
                          fill=False, edgecolor=colors[i],
                          linewidth=0.8, alpha=0.6)
            ax_left.add_patch(circle)
            circles.append(circle)
        
        # 圆环连线
        chain_line, = ax_left.plot([], [], 'b-', linewidth=0.8, alpha=0.5)
        
        # 笔尖
        pen_left, = ax_left.plot([], [], 'ro', markersize=6, zorder=10)
        pen_right, = ax_right.plot([], [], 'ro', markersize=6, zorder=10)
        
        # 轨迹线（右图）
        trace_line, = ax_right.plot([], [], 'r-', linewidth=1.8, alpha=0.9)
        
        # 信息文本
        info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        trace_x = []
        trace_y = []
        
        def update(frame):
            # t 从 0 到 1，完整绘制一圈
            t = frame / frames
            
            # 计算圆环链
            chain = self.evaluate_epicycle_chain(t)
            
            # 更新圆环位置
            for i, (ep, circle) in enumerate(zip(display_eps, circles)):
                if i < len(chain) - 1:
                    circle.center = chain[i]
            
            # 更新连线
            chain_x = [c[0] for c in chain]
            chain_y = [c[1] for c in chain]
            chain_line.set_data(chain_x, chain_y)
            
            # 笔尖位置
            final_x, final_y = chain[-1]
            pen_left.set_data([final_x], [final_y])
            pen_right.set_data([final_x], [final_y])
            
            # 累积轨迹
            trace_x.append(final_x)
            trace_y.append(final_y)
            trace_line.set_data(trace_x, trace_y)
            
            # 进度
            progress = t * 100
            info_text.set_text(f'绘制进度: {progress:.1f}%  |  '
                              f'傅里叶分量: {len(self.epicycles)}  |  '
                              f'已绘制点: {len(trace_x)}')
            
            if progress >= 99.5:
                ax_right.set_title('✓ 绘制完成！', fontsize=14,
                                  fontweight='bold', color='green')
            
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
    anim, fig = drawer.create_animation(frames=3000, interval=16,
                                        num_display_circles=50)
    
    # 4. 保存动画
    print("\n4. 保存动画...")
    try:
        anim.save('flag_fourier_drawing.gif', writer='pillow', fps=30, dpi=80)
        print("   ✓ 动画已保存为 flag_fourier_drawing.gif")
    except Exception as e:
        print(f"   保存失败: {e}")
        print("   尝试保存为 mp4...")
        try:
            anim.save('flag_fourier_drawing.mp4', writer='ffmpeg', fps=30, dpi=80)
            print("   ✓ 动画已保存为 flag_fourier_drawing.mp4")
        except Exception as e2:
            print(f"   mp4 也保存失败: {e2}")
    
    # 5. 显示
    print("\n5. 显示动画窗口...")
    plt.show()
    
    print("\n✓ 完成！")