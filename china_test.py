import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from scipy.fft import fft
import platform, os, warnings
warnings.filterwarnings('ignore')

# ============ 中文字体 ============
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

CHRED, CHYEL = '#EE1C25', '#FFFF00'


# ================================================================
# 1. 星星顶点
# ================================================================
def star_verts():
    s5 = np.sqrt(5)
    return np.array([
        [0,4],[np.sqrt(50-22*s5),-1+s5],[np.sqrt(10+2*s5),-1+s5],
        [2*np.sqrt(5-2*s5),4-2*s5],[np.sqrt(10-2*s5),-1-s5],
        [0,-6+2*s5],[-np.sqrt(10-2*s5),-1-s5],
        [-2*np.sqrt(5-2*s5),4-2*s5],[-np.sqrt(10+2*s5),-1+s5],
        [-np.sqrt(50-22*s5),-1+s5],
    ])

def xform(v, sx, sy, sc, rot):
    v = v.copy() * sc
    a = np.radians(rot)
    R = np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    v = v @ R.T
    v[:,0] += sx + 15
    v[:,1] = 10 - (v[:,1] + sy)
    return v


# ================================================================
# 2. 采样
# ================================================================
def sample_poly(verts, n):
    verts = np.vstack([verts, verts[0]])
    edges = np.linalg.norm(np.diff(verts, axis=0), axis=1)
    total = edges.sum()
    pts = []
    for i, el in enumerate(edges):
        ns = max(int(round(n * el / total)), 1)
        for j in range(ns):
            pts.append(verts[i] + j/ns * (verts[i+1] - verts[i]))
    pts = np.array(pts)
    if len(pts) != n:
        ot = np.linspace(0, 1, len(pts), endpoint=False)
        nt = np.linspace(0, 1, n, endpoint=False)
        pts = np.column_stack([np.interp(nt, ot, pts[:,0]),
                               np.interp(nt, ot, pts[:,1])])
    return pts

def sample_rect(w, h, n):
    return sample_poly(np.array([[0,0],[w,0],[w,h],[0,h]]), n)


# ================================================================
# 3. 轮廓
# ================================================================
def gen_contours(N=1500):
    ox, oy = -10, 5
    lg = xform(star_verts(), ox, oy, 3/4, 0)
    sms = []
    for dx, dy, rot in [(5,3,90+np.degrees(np.arctan(3/5))),
                         (7,1,90+np.degrees(np.arctan(1/7))),
                         (7,-2,90-np.degrees(np.arctan(2/7))),
                         (5,-4,90-np.degrees(np.arctan(4/5)))]:
        sms.append(xform(star_verts(), ox+dx, oy+dy, 1/4, rot))
    cs = [(sample_rect(30, 20, N), '#FFFFFF', '边框'),
          (sample_poly(lg, N), CHYEL, '大星')]
    for i, s in enumerate(sms):
        cs.append((sample_poly(s, N), CHYEL, f'小星{i+1}'))
    return cs, [lg] + sms


# ================================================================
# 4. 傅里叶
# ================================================================
class CFourier:
    def __init__(self, pts, mh=120):
        pts = np.asarray(pts, float)
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        n = len(pts)
        z = pts[:,0] + 1j * pts[:,1]
        c = fft(z) / n # type: ignore
        amp = np.abs(c)
        order = np.argsort(amp)[::-1]
        self.epis = []
        for idx in order:
            if amp[idx] < 1e-9:
                continue
            f = idx if idx <= n//2 else idx - n
            self.epis.append({'f': f, 'a': amp[idx], 'p': np.angle(c[idx])})
            if len(self.epis) >= mh:
                break
        self.epis.sort(key=lambda e: abs(e['f']))

    def eval_arr(self, ts):
        ts = np.asarray(ts)
        z = np.zeros(len(ts), dtype=complex)
        for e in self.epis:
            z += e['a'] * np.exp(1j * (2*np.pi*e['f']*ts + e['p']))
        return np.column_stack([z.real, z.imag]) # type: ignore

    def chain(self, t, mc=20):
        """返回所有分量的圆心链，长度 = min(mc, len(epis)) + 1"""
        cx = cy = 0.0
        pts = [(cx, cy)]
        for e in self.epis[:mc]:
            ang = 2*np.pi*e['f']*t + e['p']
            cx += e['a'] * np.cos(ang)
            cy += e['a'] * np.sin(ang)
            pts.append((cx, cy))
        return pts


# ================================================================
# 5. 静态
# ================================================================
def draw_static(sv):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-1, 31); ax.set_ylim(21, -1); ax.set_aspect('equal')
    ax.set_title('中华人民共和国国旗', fontproperties=CN_FONT, fontsize=16, fontweight='bold')
    ax.add_patch(plt.Rectangle((0,0), 30, 20, fc=CHRED, ec='black', lw=2)) # type: ignore
    for v in sv:
        ax.add_patch(Polygon(v, closed=True, fc=CHYEL, ec='gold', lw=1))
    plt.tight_layout()
    plt.savefig('flag_static.png', dpi=150, bbox_inches='tight')
    plt.show()


# ================================================================
# 6. 动画
# ================================================================
def create_anim(contours, sv,
                frames=400, fps=25,
                max_harm=120, n_circles=None):
    """n_circles=None 表示用全部分量的圆"""

    analyzers, all_traces = [], []
    for pts, _, label in contours:
        fa = CFourier(pts, max_harm)
        analyzers.append(fa)
        all_traces.append(fa.eval_arr(np.linspace(0, 1, frames, endpoint=False)))
        print(f"  [{label}] {len(fa.epis)} 分量")

    # 每个轮廓实际使用的圆环数 = 全部分量
    nc_list = []
    for fa in analyzers:
        nc = len(fa.epis) if n_circles is None else min(n_circles, len(fa.epis))
        nc_list.append(nc)
        print(f"    → 圆环数: {nc}")

    ap = np.vstack(all_traces); mg = 2
    xn, xx = ap[:,0].min()-mg, ap[:,0].max()+mg
    yn, yx = ap[:,1].min()-mg, ap[:,1].max()+mg

    fill_c = [CHRED, CHYEL, CHYEL, CHYEL, CHYEL, CHYEL]

    # 深色系 —— 白底上清晰可见
    circ_colors = ['#333333', '#B8860B', '#CC6600', '#CC2222', '#BB2288', '#8822AA']
    link_colors = ['#222222', '#996600', '#AA5500', '#AA1111', '#991177', '#771199']
    trac_colors = ['#555555', '#DAA520', '#E08800', '#DD4444', '#DD3399', '#AA44BB']
    pen_colors  = ['#FF0000', '#FF8C00', '#FF6600', '#FF2222', '#FF33AA', '#CC44FF']

    # --- 双面板 ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    for ax in [axL, axR]:
        ax.set_xlim(xn, xx); ax.set_ylim(yx, yn); ax.set_aspect('equal')
        ax.set_facecolor('white'); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    axL.set_title('傅里叶圆环机构', fontproperties=CN_FONT, fontsize=12,
                  fontweight='bold', color='#333')
    axR.set_title('渐进绘制 & 填色', fontproperties=CN_FONT, fontsize=12,
                  fontweight='bold', color='#333')

    # 右图淡参考线
    for pts, color, _ in contours:
        c = '#ddd' if color == '#FFFFFF' else '#FFF3B0'
        axR.plot(pts[:,0], pts[:,1], '--', color=c, alpha=0.3, lw=0.5)

    # --- 创建绘图对象 ---
    circle_objs = []
    chain_lines = []
    traceL = []
    penL = []
    traceR = []
    penR = []
    fills = []

    for k, fa in enumerate(analyzers):
        nc = nc_list[k]

        # 左图：所有圆环
        cks = []
        for i in range(nc):
            # 小圆用更细更透明，大圆用粗一点
            r = fa.epis[i]['a']
            if r > 0.5:
                lw, alpha = 0.8, 0.45
            elif r > 0.1:
                lw, alpha = 0.5, 0.3
            else:
                lw, alpha = 0.3, 0.15
            ci = Circle((0, 0), r, fill=False,
                        ec=circ_colors[k], lw=lw, alpha=alpha, zorder=3)
            axL.add_patch(ci)
            cks.append(ci)
        circle_objs.append(cks)

        # 左图：连杆线
        cl, = axL.plot([], [], '-', color=link_colors[k],
                       lw=0.8, alpha=0.5, zorder=5)
        chain_lines.append(cl)

        # 左图：已画轨迹
        tl_l, = axL.plot([], [], '-', color=trac_colors[k],
                         lw=1.0, alpha=0.6, zorder=6)
        traceL.append(tl_l)

        # 左图：笔尖
        pl, = axL.plot([], [], 'o', color=pen_colors[k],
                       ms=5, zorder=10, markeredgecolor='black',
                       markeredgewidth=0.5)
        penL.append(pl)

        # 右图：轨迹
        lw_r = 1.4 if k <= 1 else 1.0
        tl_r, = axR.plot([], [], '-', color=trac_colors[k],
                         lw=lw_r, alpha=0.85, zorder=12)
        traceR.append(tl_r)

        # 右图：笔尖
        pr, = axR.plot([], [], 'o', color=pen_colors[k],
                       ms=4, zorder=15, markeredgecolor='black',
                       markeredgewidth=0.5)
        penR.append(pr)

        # 右图：填色
        fp = Polygon([[0,0]], closed=True, fc=fill_c[k],
                     ec='none', alpha=0.0, zorder=5)
        axR.add_patch(fp)
        fills.append(fp)

    info = fig.text(0.5, 0.02, '', ha='center', fontproperties=CN_FONT,
                    fontsize=9, color='#555',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#f5f5f5', ec='#ddd'))

    # 图例
    axR.legend(handles=[Line2D([0],[0], color=trac_colors[k], lw=2,
               label=contours[k][2]) for k in range(len(contours))],
               loc='upper right', fontsize=8, facecolor='white',
               edgecolor='#ccc', labelcolor='#333', prop=CN_FONT, framealpha=0.8)

    def thin(arr, mx=250):
        if len(arr) <= mx:
            return arr
        return arr[np.linspace(0, len(arr)-1, mx, dtype=int)]

    def update(frame):
        t = frame / frames
        n = frame + 1
        is_last = (frame == frames - 1)

        for k, fa in enumerate(analyzers):
            seg = all_traces[k][:n]
            px, py = seg[-1]
            nc = nc_list[k]

            if not is_last:
                # === 左图：圆环跟随 ===
                ch = fa.chain(t, nc)
                for i in range(nc):
                    if i < len(ch) - 1:
                        circle_objs[k][i].center = ch[i]
                        circle_objs[k][i].set_visible(True)

                # 连杆线
                xs = [p[0] for p in ch]
                ys = [p[1] for p in ch]
                chain_lines[k].set_data(xs, ys)

                # 左图轨迹
                traceL[k].set_data(seg[:,0], seg[:,1])

                # 左图笔尖
                penL[k].set_data([px], [py])

                # === 右图 ===
                traceR[k].set_data(seg[:,0], seg[:,1])
                penR[k].set_data([px], [py])

                # 填色
                if t > 0.08 and n >= 3:
                    alpha = min(1.0, ((t - 0.08) / 0.82) ** 1.3)
                    fills[k].set_xy(thin(seg))
                    fills[k].set_alpha(alpha)
            else:
                # === 最后一帧 ===
                full = all_traces[k]

                # 左图：隐藏圆环和连杆，保留完整轨迹
                for ci in circle_objs[k]:
                    ci.set_visible(False)
                chain_lines[k].set_data([], [])
                penL[k].set_data([], [])

                # 左图保留完整轨迹线
                traceL[k].set_data(full[:,0], full[:,1])
                traceL[k].set_alpha(1.0)
                traceL[k].set_linewidth(1.5 if k <= 1 else 1.0)

                # 右图完整填色
                fills[k].set_xy(full)
                fills[k].set_alpha(1.0)
                traceR[k].set_data(full[:,0], full[:,1])
                penR[k].set_data([], [])

        pct = t * 100
        if not is_last:
            info.set_text(f'进度 {pct:.0f}%  |  {max_harm} 谐波  |  圆环=全部分量')
        else:
            info.set_text('✓ 完成')
            axL.set_title('傅里叶绘制结果', fontproperties=CN_FONT,
                          fontsize=12, fontweight='bold', color='#228B22')
            axR.set_title('五星红旗 ✓', fontproperties=CN_FONT,
                          fontsize=12, fontweight='bold', color='#228B22')

    anim = FuncAnimation(fig, update, frames=frames, # type: ignore
                         interval=1000//fps, blit=False, repeat=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.07)
    return anim, fig, fps


# ================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  五星红旗 · 傅里叶双面板 · 全圆环版")
    print("=" * 50)

    contours, sv = gen_contours(N=1500)
    draw_static(sv)

    print("\n构建动画 (400帧 ≈ 16秒)...")
    anim, fig, fps = create_anim(contours, sv,
                                  frames=400, fps=25,
                                  max_harm=120,
                                  n_circles=None)  # None = 全部分量都画圆

    print("保存 GIF...")
    try:
        anim.save('flag_fourier.gif', writer='pillow', fps=fps, dpi=72)
        mb = os.path.getsize('flag_fourier.gif') / 1024 / 1024
        print(f"  ✓ flag_fourier.gif  ({mb:.1f} MB)")
    except Exception as e:
        print(f"  GIF失败: {e}")
        try:
            anim.save('flag_fourier.mp4', writer='ffmpeg', fps=fps, dpi=80)
            print("  ✓ flag_fourier.mp4")
        except Exception as e2:
            print(f"  MP4也失败: {e2}")

    plt.show()
    print("✓ 完成")