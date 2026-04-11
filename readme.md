# 傅里叶级数绘制五星红旗

使用傅里叶级数（Fast Fourier Transform）动画绘制中华人民共和国国旗。

## 项目原理（通俗版）

想象一下：如果你有一堆积木，每块积木以不同的速度转圈，把它们叠在一起，它们的末端会画出一条曲线。

**傅里叶级数就是这种"转圈积木"的数学表达**：
- 每一个"积木"是一个圆，以固定速度旋转
- 圆的大小（振幅）决定它能画多远的距离
- 旋转的速度（频率）决定它多久转一圈
- 起始角度（相位）决定它从哪里开始

如果我们把国旗轮廓上的所有点收集起来，用傅里叶分析分解，就能得到几百个这样的"圆"。让它们同时旋转，它们画出的叠加轨迹就是五星红旗！

## 运行依赖

```bash
pip install numpy matplotlib scipy
```

## 使用方法

```bash
python china.py          # 基础版本：整个国旗作为一条曲线
python china_test.py    # 多路版本：6条独立曲线同时绘制
python china_tes_color.py  # 填色版本：带渐进填色效果
```

---

## 详细代码解释

### china.py - 基础版本

这个文件把整个国旗（矩形边框 + 5颗星）当作**一条连续的曲线**来处理。

#### 第1-8行：导入必要的库

```python
import numpy as np              # 用于数学计算（数组、矩阵、三角函数等）
import matplotlib.pyplot as plt # 用于绘图（画图、做动画）
from matplotlib.animation import FuncAnimation  # 用于制作动画
from matplotlib.patches import Circle, Polygon  # 用于画圆和多边形
from matplotlib.font_manager import FontProperties  # 用于设置中文字体
from scipy.fft import fft       # 用于傅里叶变换（核心算法）
import warnings
warnings.filterwarnings('ignore')  # 忽略一些烦人的警告信息
```

**通俗解释**：就像做菜需要锅碗瓢盆一样，这段代码引入了各种"工具"，后面会用它们来完成画图和计算的工作。

#### 第10-15行：设置颜色和字体

```python
# macOS 中文字体（黑体）
CN_FONT = FontProperties(fname='/System/Library/Fonts/STHeiti Medium.ttc')

# 国旗的两种颜色（红色和黄色）
CHRED = '#EE1C25'   # 中国红
CHYEL = '#FFFF00'   # 金黄色
```

**通俗解释**：定义好要用的颜色，就跟画画前准备颜料一样。

#### 第18-41行：创建五角星顶点 `create_star_vertices_latex`

```python
def create_star_vertices_latex(scale=1.0):
    """
    根据 LaTeX 模板创建五角星的10个顶点坐标
    这是一个数学上精确的五角星形状
    """
    s5 = np.sqrt(5)  # 计算 sqrt(5)，后面会反复用到
    # 五角星的10个顶点坐标（基于数学公式计算）
    vertices = np.array([
        [0, 4],
        [np.sqrt(50 - 22*s5), -1 + s5],
        [np.sqrt(10 + 2*s5), -1 + s5],
        ...
    ])
    return vertices * scale  # 可以缩放大小
```

**通俗解释**：这个函数用数学公式算出五角星的10个顶点位置。这就像用尺子和量角器画出一个标准的五角星。

#### 第44-72行：坐标变换 `transform_star_to_flag`

```python
def transform_star_to_flag(vertices, shift_x, shift_y, scale, rotate_deg):
    """
    把五角星放到国旗的准确位置
    步骤：缩放 → 旋转 → 移动 → 转换坐标系
    """
    # 1. 缩放（改变大小）
    v = vertices * scale

    # 2. 旋转
    angle_rad = np.radians(rotate_deg)  # 把角度转成弧度
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    v = v @ rot_matrix.T  # 矩阵乘法实现旋转

    # 3. 平移（移动位置）
    v[:, 0] += shift_x
    v[:, 1] += shift_y

    # 4. 坐标系转换：LaTeX坐标系 → 屏幕坐标系
    # LaTeX: 原点在中心，y向上
    # 屏幕: 原点在左上角，y向下
    v[:, 0] = v[:, 0] + 15      # x轴向右平移15
    v[:, 1] = 10 - v[:, 1      # y轴翻转）
    return v
```

**通俗解释**：就像把一张五角星贴纸贴到国旗上，需要调整大小、旋转角度、放到正确位置。这里处理的就是这件事。

#### 第75-85行：沿多边形边缘采样 `interpolate_polygon_edges`

```python
def interpolate_polygon_edges(vertices, num_points_per_edge):
    """沿多边形的每条边均匀地取点"""
    points = []
    n = len(vertices)  # 顶点数量
    for i in range(n):
        start = vertices[i]           # 边的起点
        end = vertices[(i + 1) % n]   # 边的终点（%n表示回到起点）
        # 在起点和终点之间插入多个点
        for j in range(num_points_per_edge):
            t = j / num_points_per_edge  # 比例（0到1之间）
            points.append(start + t * (end - start))  # 线性插值
    return np.array(points)
```

**通俗解释**：假设五角星有一条边从A点到B点，这个函数在A到B之间均匀地取出一系列点。这样我们就能把"形状"变成"一系列坐标点"。

#### 第88-194行：生成国旗轮廓 `get_flag_outline`

这是最复杂的函数，核心思路是：
1. 计算5颗星的位置和形状
2. 画一个30×20的矩形（国旗的边界）
3. **把这些东西连成一条连续的曲线**

```python
def get_flag_outline(num_samples=6000):
    """
    获取五星红旗轮廓点，形成一条连续的闭合曲线
    """
    # 1. 计算5颗星的位置（使用前面定义的函数）
    large_star = transform_star_to_flag(...)  # 大星
    small_star1 = transform_star_to_flag(...) # 小星1
    small_star2 = transform_star_to_flag(...) # 小星2
    small_star3 = transform_star_to_flag(...) # 小星3
    small_star4 = transform_star_to_flag(...) # 小星4

    # 2. 画矩形边框
    rect_vertices = np.array([
        [0, 0], [width, 0], [width, height], [0, height]
    ])

    # 3. 把所有东西连成一条线
    # 顺序：矩形 → 过渡线 → 大星 → 过渡线 → 小星1 → ... → 回到矩形
    all_points = []
    all_points.append(rect_pts)          # 矩形边缘的点
    all_points.append(trans1)            # 过渡到第一颗星
    all_points.append(large_pts)         # 大星的边缘
    all_points.append(trans2)            # 过渡到第二颗星
    all_points.append(small_star1_pts)   # 小星的边缘
    ...

    return points, all_stars
```

**通俗解释**：想象用一个细细的画笔沿着国旗的轮廓画一圈，从左上角开始，沿边框走，画大五角星，画四颗小五角星，最后回到起点。这个函数就是把这条"画过的路径"变成8000个坐标点。

#### 第197-236行：绘制静态国旗 `draw_static_flag`

```python
def draw_static_flag(all_stars):
    """绘制静态国旗图片"""
    fig, ax = plt.subplots(figsize=(12, 8))  # 创建画布

    # 画红色背景
    flag_bg = plt.Rectangle((0, 0), 30, 20,
                            facecolor=CHRED, edgecolor='black', linewidth=2)
    ax.add_patch(flag_bg)

    # 画5颗黄色星星
    for star in all_stars:
        star_poly = Polygon(star, facecolor=CHYEL, ...)
        ax.add_patch(star_poly)

    plt.savefig('chinese_flag_correct.png', dpi=150)  # 保存图片
    plt.show()  # 显示窗口
```

**通俗解释**：这就是画一个静态的国旗图片，给后续动画做参考用。

#### 第239-417行：傅里叶动画类 `FourierDrawingAnimation`

这是核心部分！把曲线变成动画的魔法就在这里。

##### 初始化（第242-251行）

```python
class FourierDrawingAnimation:
    """傅里叶级数绘制动画"""

    def __init__(self, points, max_harmonics=150):
        self.points = np.asarray(points, dtype=float)
        self.max_harmonics = max_harmonics  # 最多使用多少个"圆"

        # 确保曲线首尾相连（闭合）
        if not np.allclose(self.points[0], self.points[-1]):
            self.points = np.vstack([self.points, self.points[0]])

        self.compute_fourier_coefficients()  # 计算傅里叶系数
        self.prepare_epicycles()             # 准备圆环数据
```

**通俗解释**：这个"类"就像一个机器，把输入的8000个点变成一组旋转的圆环。

##### 计算傅里叶系数（第253-258行）

```python
def compute_fourier_coefficients(self):
    """计算傅里叶系数"""
    n_points = len(self.points)
    # 把 (x, y) 坐标转成复数 x + yj
    z = self.points[:, 0] + 1j * self.points[:, 1]
    # 用 FFT 快速计算傅里叶系数
    self.coeffs = fft(z) / n_points
```

**通俗解释**：FFT是一种算法，能把一堆点"分解"成一系列不同频率的圆周运动。这里的"系数"就是描述每个圆的大小、速度、起始角度。

##### 准备圆环数据（第260-293行）

```python
def prepare_epicycles(self):
    """准备圆环数据，按幅度（大小）排序"""
    amplitudes = np.abs(self.coeffs)  # 取出每个分量的"大小"
    indices = np.argsort(amplitudes)[::-1]  # 从大到小排序

    self.epicycles = []
    for idx in indices:
        amp = amplitudes[idx]
        if amp < 1e-8:  # 太小的不要（噪音）
            continue
        # 计算频率和相位
        if idx <= n // 2:
            freq = idx
        else:
            freq = idx - n
        phase = np.angle(self.coeffs[idx])

        # 存到列表里
        self.epicycles.append({
            'freq': freq,      # 旋转速度
            'amp': amp,        # 圆的大小
            'phase': phase,   # 起始角度
        })

    # 按频率排序（为了让动画更流畅）
    self.epicycles.sort(key=lambda x: abs(x['freq']))
```

**通俗解释**：这一步把傅里叶系数整理成一个个"圆环"的数据。每个圆环有：大小、速度、起始角度。取前200个最大的圆环，忽略太小的噪音。

##### 计算某时刻的位置（第295-301行）

```python
def evaluate(self, t):
    """在参数 t (0~1) 处计算傅里叶级数的值"""
    z = 0 + 0j  # 从原点开始
    for ep in self.epicycles:
        # 计算这个圆在该时刻的位置
        angle = 2 * np.pi * ep['freq'] * t + ep['phase']
        # 叠加到总位置
        z += ep['amp'] * np.exp(1j * angle)
    return z.real, z.imag  # 返回 x, y 坐标
```

**通俗解释**：当时间 t=0.5（一半）时，所有圆环都转到对应位置，把它们加在一起，就是画笔当前的位置。

##### 动画制作（第318-417行）

```python
def create_animation(self, frames=3000, interval=20, num_display_circles=40):
    """创建绘制动画"""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 9))
    # ... 设置画布 ...

    # 左图：显示圆环机构
    # 右图：显示画出的轨迹
```

动画的核心是 `update` 函数，每一帧都会调用一次：

```python
def update(frame):
    t = frame / frames  # 时间进度（0到1）

    # 1. 计算所有圆环的位置
    chain = self.evaluate_epicycle_chain(t)

    # 2. 移动圆环到新位置
    for i, (ep, circle) in enumerate(zip(display_eps, circles)):
        if i < len(chain) - 1:
            circle.center = chain[i]

    # 3. 画笔尖位置
    final_x, final_y = chain[-1]

    # 4. 记录轨迹
    trace_x.append(final_x)
    trace_y.append(final_y)

    # 5. 更新画面
    ...
```

**通俗解释**：动画有3000帧，每一帧让圆环转动一点点，然后把画笔位置记录下来，连成线。第1帧画一点，第100帧画更多...直到第3000帧画完整条线。

#### 第420-461行：主程序入口

```python
if __name__ == "__main__":
    print("傅里叶级数绘制五星红旗")

    # 1. 生成8000个轮廓点
    points, all_stars = get_flag_outline(num_samples=8000)

    # 2. 画静态图
    draw_static_flag(all_stars)

    # 3. 创建动画
    drawer = FourierDrawingAnimation(points, max_harmonics=200)
    anim, fig = drawer.create_animation(frames=3000)

    # 4. 保存为 GIF
    anim.save('flag_fourier_drawing.gif', writer='pillow', fps=30)

    # 5. 显示
    plt.show()
```

---

### china_test.py - 多路版本

这个版本把**6条独立的轮廓**分别做傅里叶分析，同时绘制。

**与 china.py 的区别**：
- china.py：把整个国旗当作一条线，会有"跨图形"的过渡线
- china_test.py：矩形是矩形，星星是星星，6条独立的线同时画

```python
# 6条独立的轮廓
contours = [
    (rect_pts, '白色', '矩形边框'),
    (large_pts, '黄色', '大星'),
    (small1_pts, '黄色', '小星1'),
    ...
]

# 为每条轮廓创建独立的傅里叶分析器
analyzers = []
for pts, color, label in contours:
    fa = SingleContourFourier(pts, max_harmonics=200, label=label)
    analyzers.append(fa)

# 同时驱动6个分析器
def update(frame):
    for fa in analyzers:
        # 每个分析器独立计算自己的圆环和轨迹
        ...
```

**通俗解释**：就像有6个人同时画6幅画，每个人只画自己负责的部分（矩形、或者某一颗星）。这样不会有跨图形的连线，更加干净。

---

### china_tes_color.py - 填色版本

在 china_test.py 基础上增加了**渐进填色**效果。

```python
# 右图：画完的轨迹逐渐填充颜色
fill_patches = []  # 6个填充多边形

def update(frame):
    ...
    # 随时间逐渐增加透明度
    if t > fill_start_frac and n_pts >= 3:
        alpha = (t - fill_start_frac) / (fill_full_frac - fill_start_frac)
        fill_patches[k].set_alpha(alpha)  # 越来越不透明
    ...
```

**通俗解释**：当画笔经过的地方，慢慢填充上正确的颜色（红色背景、黄色星星）。就像画填色书一样，边画边填充。

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `chinese_flag_correct.png` | 静态国旗图片 |
| `flag_fourier_drawing.gif` | 基础版动画（左：圆环机构，右：绘制轨迹） |
| `flag_6fourier.gif` | 多路版动画 |
| `flag_6fourier_fill.gif` | 填色版动画 |

---

## 核心概念总结

1. **傅里叶级数**：把任意形状分解成一系列圆周运动的叠加
2. **FFT（快速傅里叶变换）**：高效计算傅里叶系数的算法
3. **圆环机构**：每个傅里叶分量对应一个旋转的圆
4. **动画原理**：让圆环随时间旋转，记录画笔轨迹