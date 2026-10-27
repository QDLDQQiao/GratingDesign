*Thinking...*

> The user needs a technical specification document in Chinese for implementing a simulation program that validates the SB-BSTG grating design—it should be parametrized by X-ray energy, detector specifications, and distances, include grating structure design and pattern analysis with tolerance studies, and outline next steps for signal analysis and reconstruction methods. This spec is meant to be handed to another AI coding tool for implementation.
> 
> I'm deciding on the format: the system prompt discourages bullet points for technical documentation, but since this is a specification for code generation, I'll use structured headings, prose paragraphs, equations in LaTeX notation, code blocks for module signatures and pseudocode, and tables for parameter specifications where appropriate. I'll minimize bullets and keep the focus on clear technical prose with supporting structured elements.
> 
> Now I'm laying out the physical and numerical specifications: units convention, sampling strategy with band-limited angular spectrum to prevent aliasing, material data sourcing from libraries like xraylib or embedded tables, grating generation with star carrier patterns and blue-noise binarization, propagation modeling that handles both polychromatic light as incoherent energy-weighted sums and partial coherence through source blur convolution, and detector simulation including pixel binning, modulation transfer function, and Poisson noise.
> 
> The analysis module will compute Fourier-space visibility metrics and local demodulation, build crosstalk matrices, trace Talbot carpet patterns across propagation distance, and evaluate energy tolerance by sweeping the spectrum and combining results into joint maps.
> 
> For optimization, I'm planning a two-stage workflow: first validate the forward model, then run parameter sweeps followed by optional gradient-based or level-set optimization depending on whether the grating parameters are continuous or binary.
> 
> Validation includes unit tests for known cases like plane grating Talbot distances and π-grating fractional Talbot positions, plus convergence checks. Outputs go to HDF5 with accompanying plots and a JSON report flagging go/no-go based on visibility thresholds, all deployable via Python with optional GPU acceleration and a YAML config interface.
> 
> Next I'm building the signal analysis and reconstruction pipeline: Fourier demodulation with windowed extraction and phase differences to compute differential phase contrast per direction, dark-field analysis with anisotropic fitting, speckle tracking fusion via cross-correlation, and absolute wavefront recovery by comparing against a numerically simulated reference. I'll validate against phantom tests with known Zernike modes and anisotropic structures, computing RMSE and comparing against Cramér-Rao bounds.
> 
> For the specification document, I'm including configuration examples, module API signatures, and pseudocode for the collision checker, band-limited angular spectrum method, and visibility extraction. I'll cover the key physics formulas—phase depth, transfer functions for both ASM and Fresnel propagation, source blur scaling for parallel and cone geometries, and photon noise—plus sampling requirements ensuring the carrier frequencies align with FFT grid points and domain covers multiple supercells with periodic boundaries. I'm aiming for a detailed but readable spec in Chinese with proper mathematical notation.

# SB-BSTG 模拟验证：部署规格书（可直接交给代码生成 AI）

本文档分为四部分：A. 系统架构与部署环境；B. 各模块的详细规格（含物理模型、公式、伪代码、验收测试）；C. 容忍度分析流程（能量、距离）；D. 下一步计划（信号分析与重建）。你可以把 B 部分逐模块拆分成独立任务喂给代码生成 AI，每个模块都附有单元测试标准，便于验证 AI 生成代码的正确性。

---

## A. 系统架构与部署环境

### A.1 技术栈

推荐 Python 3.11+，核心依赖为 numpy、scipy、matplotlib、h5py、pyyaml；可选依赖为 cupy（GPU 加速 FFT，网格大于 \(8192^2\) 时必要）与 xraylib（材料光学常数；若不可用则使用内置 δ/β 表，见 B.2）。禁止引入重量级仿真框架，全部传播代码自实现，以保证可控与可审计。

### A.2 目录结构

```
sbbstg-sim/
├── config/
│   ├── default.yaml          # 主配置
│   └── materials.yaml        # 内置 δ/β 表（备用）
├── src/
│   ├── config.py             # 配置解析与单位换算
│   ├── materials.py          # δ(E), β(E)
│   ├── grating.py            # 光栅生成器（星形载频 + 蓝噪声 + 二值化）
│   ├── collision.py          # 互调碰撞检查器
│   ├── propagate.py          # 带限角谱传播（单色相干核）
│   ├── system.py             # 部分相干 + 多色 + 几何放大的正向模型
│   ├── detector.py           # 像素积分、MTF、噪声
│   ├── analysis.py           # 频谱分析、可见度、串扰
│   ├── scans.py              # z 扫描、E 扫描、二维容忍度图
│   └── report.py             # go/no-go 判定与图表输出
├── tests/                    # 每模块的单元测试（见各节验收标准）
├── scripts/
│   ├── run_design.py         # 生成光栅并做碰撞检查
│   ├── run_forward.py        # 单点正向模拟
│   └── run_tolerance.py      # 容忍度扫描
└── results/                  # HDF5 + PNG 输出
```

### A.3 全局约定

内部单位统一为米、电子伏特、弧度。所有二维数组约定 axis 0 为 \(y\)、axis 1 为 \(x\)。FFT 一律使用 `np.fft.fft2` + `fftshift` 的中心化约定，频率轴由 `fftfreq(N, d=Δx)` 生成。**关键采样约定：模拟窗口必须为超晶胞 \(P\) 的整数倍，且采样数使所有载频 \(\mathbf f_j\) 严格落在 FFT 网格点上**，这样可用周期边界条件、无窗函数泄漏，可见度测量才是干净的。

### A.4 主配置文件示例

```yaml
# config/default.yaml
source:
  type: parallel            # parallel | cone
  energy_keV: 12.4          # 中心能量
  bandwidth: 0.0            # ΔE/E (FWHM)；0 表示单色
  n_energy_samples: 7       # 多色时的能量采样数
  size_um: 0.0              # 源尺寸 FWHM；parallel 模式下为等效模糊
  distance_m: 10.0          # 源—光栅距离 L（cone 模式使用）

grating:
  supercell_um: 56.0        # 超晶胞 P
  m: 7                      # 轴向载频指数 (±7,0),(0,±7)
  n: 5                      # 斜向载频指数 (±5,±5)
  amplitudes: [1.0, 1.0, 1.0, 1.0]   # a_j，探测面均衡后回填
  phases_deg: [0, 0, 0, 0]  # ψ_j
  phase_levels: 2           # 2 = 二值 {0,π}；4 = 四级
  fill_factor: 0.5          # 二值化阈值 τ 由此反解
  blue_noise:
    enabled: false          # 第一阶段关闭
    epsilon: 0.1
    feature_um: 2.0         # 最小特征尺寸
  material: Si
  substrate: none

geometry:
  z_m: 0.080                # 光栅—探测器距离（单点模式）

detector:
  pixel_um: 1.0             # 有效像素
  aperture_fill: 1.0        # 像素孔径填充因子
  mtf_model: sinc_gauss     # sinc(像素孔径) × 高斯附加模糊
  mtf_sigma_um: 0.5
  photons_per_pixel: 1.0e4  # 0 表示无噪声
  n_noise_realizations: 10

simulation:
  oversample: 16            # 每个最细特征的采样点数下限
  n_supercells: 4           # 窗口 = n_supercells × P
  use_gpu: auto

scan:
  z_range_m: [0.01, 0.30]
  z_steps: 120
  energy_range_keV: [10.0, 16.0]
  energy_steps: 25
```

---

## B. 模块规格

### B.1 config.py

解析 YAML，换算单位，派生量包括：波长 \(\lambda = hc/E\)（\(hc = 1.23984198\ \mathrm{keV\cdot nm}\)）；采样间隔 \(\Delta x\)（由 oversample 与最细特征决定，并向下取整使 \(P/\Delta x\) 为整数）；总网格数 \(N = n_{\rm supercells} \times P/\Delta x\)。校验规则：\(N\) 应为 2 的幂或高合成数；\(\Delta x \leq p_{\min}/16\)，其中 \(p_{\min} = P/(\sqrt{2}\,n)\)；若蓝噪声开启，\(\Delta x \leq \mathrm{feature\_um}/8\)。

### B.2 materials.py

接口为 `delta_beta(material: str, E_keV: float) -> (delta, beta)`。优先调用 xraylib；不可用时使用内置表并做对数插值。内置表至少覆盖 Si、Au、C(diamond)、PMMA、Ni，能量范围 5–30 keV。派生函数 `pi_height(material, E)` 返回

$$
h_\pi(E) = \frac{\lambda}{2\,\delta(E)}
$$

以及复透过率构造函数：厚度图 \(t(x,y)\) 对应

$$
T(x,y) = \exp\!\left[-\frac{2\pi}{\lambda}\beta\, t(x,y)\right]\exp\!\left[-i\frac{2\pi}{\lambda}\delta\, t(x,y)\right]
$$

**验收测试**：Si 在 12.4 keV 处 \(\delta\) 与 CXRO 表值偏差小于 2%；\(h_\pi\) 随能量近似线性增长（远离吸收边时 \(\delta \propto E^{-2}\)，故 \(h_\pi \propto E\)，由 \(\lambda \propto E^{-1}\) 得出）。

### B.3 grating.py（光栅生成器）

生成流程分四步。

第一步构造连续设计场。在一个 \(P \times P\) 超晶胞上（随后平铺）：

$$
M_0(x,y) = \sum_{j=1}^{4} a_j \cos(2\pi \mathbf f_j \cdot \mathbf r + \psi_j) + \epsilon\, B_{\rm BN}(x,y)
$$

其中载频集合为

$$
\mathbf f_j \in \frac{1}{P}\{(m,0),\ (0,m),\ (n,n),\ (n,-n)\},\qquad m=7,\ n=5
$$

第二步二值化。阈值 \(\tau\) 由目标填充因子反解：对 \(M_0\) 的直方图取分位数，使 \(\Phi=\pi\) 区域占比等于 `fill_factor`。输出厚度图 \(t(x,y) = h_\pi \cdot \mathbb{1}[M_0 \geq \tau]\)。四级相位模式下将 \(M_0\) 按四分位量化到 \(\{0, h_\pi/2, h_\pi, 3h_\pi/2\}\)（注意 \(h_{\pi/2} = h_\pi/2\) 仅在忽略吸收色散时成立，代码中直接按目标相位反解厚度）。

第三步蓝噪声生成（第一阶段可跳过，但接口必须预留）。使用 void-and-cluster 算法在超晶胞上生成周期性二值蓝噪声点阵，特征尺寸受 `feature_um` 约束；对其做高斯低通得到连续 \(B_{\rm BN}\)，再在傅里叶域**挖除所有载频窗口 ± guard band**（guard band 半径取载频间最小距离的 30%），保证 \(B_{\rm BN}\) 谱不侵入解调窗口。

第四步加工规则检查。形态学操作检测：小于最小特征尺寸的孤岛、小于最小间隙的狭缝、尖角（可选倒角）。违规区域自动修补（开闭运算）并记录修补比例；修补比例超过 5% 时告警。

**验收测试**：（1）关闭蓝噪声、fill_factor=0.5 时，掩模谱在 \((\pm 7,0),(0,\pm 7),(\pm 5,\pm 5)\) 处出现主峰；（2）实际填充因子与目标偏差小于 0.5%；（3）图案严格 \(P\)-周期（平移 \(P\) 后逐点相等）。

### B.4 collision.py（互调碰撞检查器，必做）

二值化是强非线性操作，掩模谱包含所有整数组合频率

$$
\mathbf f_{\rm IM} = \sum_j k_j \mathbf f_j,\qquad k_j \in \mathbb{Z},\ \ \sum_j |k_j| \leq K_{\max}
$$

取 \(K_{\max}=3\)（覆盖二阶、三阶互调）。检查器做两件事。

其一是**解析枚举**：生成全部 \(\mathbf f_{\rm IM}\) 指数，检测是否落入任一解调窗口（四个载频窗口 + DC 窗口，各含 guard band）。例如 \(m=7,n=5\) 时 \((7,0)-(5,5)=(2,-5)\)，模长 \(\sqrt{29}/P \approx 5.39/P\)，需确认它与最近载频（模长 \(7/P\) 或 \(7.07/P\)）的距离大于 guard band。输出碰撞清单与安全裕量表。

其二是**数值审计**：对实际二值化后的掩模谱，测量每个解调窗口内**非目标峰**的总能量与目标峰能量之比，定义污染度

$$
\rho_j = \frac{\int_{\Omega_j} |\hat T|^2\, d\mathbf f - |\hat T(\mathbf f_j)|^2}{|\hat T(\mathbf f_j)|^2}
$$

**验收标准**：所有 \(\rho_j < 0.02\)（-17 dB）。超标则报告最强污染项的 \((k_1,k_2,k_3,k_4)\) 组合，供调整 \((m,n)\) 或 \(a_j\)。

伪代码：

```
def audit_intermodulation(T_hat, carriers, guard, K_max=3):
    im_list = enumerate_combinations(carriers, K_max)   # 全部 Σk_j f_j
    collisions = [f for f in im_list
                  if min(dist(f, c) for c in carriers + [DC]) < guard
                  and f not in carriers]
    rho = {}
    for j, fj in enumerate(carriers):
        win = disk_mask(fj, guard)
        E_total = sum(|T_hat[win]|^2)
        E_peak  = |T_hat[fj]|^2
        rho[j] = (E_total - E_peak) / E_peak
    return collisions, rho
```

### B.5 propagate.py（单色相干传播核）

使用带限角谱法（band-limited ASM，参照 Matsushima & Shimobaba 的带限准则）：

$$
U_z = \mathcal F^{-1}\!\left\{ \mathcal F[U_0]\cdot H(f_x,f_y)\cdot W_{\rm BL}(f_x,f_y) \right\}
$$

$$
H(f_x,f_y) = \exp\!\left[i k z \sqrt{1 - \lambda^2(f_x^2+f_y^2)}\right],\qquad k = \frac{2\pi}{\lambda}
$$

\(W_{\rm BL}\) 为带限窗，截止频率 \(f_{\rm limit} = \left(\lambda\sqrt{(2 z f_{\rm s})^{-2}+1}\right)^{-1}\)，其中 \(f_s\) 由窗口尺寸决定；由于本模拟采用严格周期边界（光栅周期平铺、无孤立孔径），带限主要防范数值混叠而非窗口截断，仍必须实现。X 射线情形 \(\lambda^2 f^2 \ll 1\)，允许退化为 Fresnel 核 \(\exp[-i\pi\lambda z(f_x^2+f_y^2)]\)，但保留完整核作为默认。

**验收测试**（这一组测试是整个流水线正确性的锚点，务必要求代码 AI 全部通过）：

1. **能量守恒**：无吸收纯相位物体传播后总强度守恒，相对误差 \(<10^{-10}\)。
2. **一维 π 相位光栅分数 Talbot 自检**：周期 \(p\) 的二值 π 相位光栅，在 \(z = p^2/(8\lambda)\) 处强度图样应为周期 \(p/2\) 的高对比条纹，可见度 \(>0.95\)（理想相干、无探测器模糊时）；在 \(z = p^2/(4\lambda)\) 附近可见度接近零。
3. **吸收光栅整数 Talbot 距离**：\(z_T = 2p^2/\lambda\) 处自成像，与 \(z=0\) 图样的归一化互相关 \(>0.999\)。
4. **采样收敛**：oversample 从 16 提高到 32，任一可见度 \(V_j\) 变化 \(<0.5\%\)。

### B.6 system.py（完整正向模型：多色 + 部分相干 + 几何）

正向模型是对相干核的三重加权求和/卷积。

**多色**：探测强度为能量谱加权的非相干叠加

$$
I(x,y;z) = \int S(E)\, \eta_{\rm det}(E)\, I_{\rm coh}(x,y;z,E)\, dE \approx \sum_{k} w_k\, I_{\rm coh}(x,y;z,E_k)
$$

能量采样用 Gauss-Legendre 节点覆盖 \(E_0(1 \pm \Delta)\)，\(n\) 由配置给定（默认 7）。注意每个 \(E_k\) 下光栅相移不同（厚度固定、\(\delta(E_k)\) 变化），透过率 \(T\) 须逐能量重算——这正是能量容忍度的物理来源，不可用"固定 π 相移"偷懒。

**部分相干（源模糊）**：平行束模式下，有限源尺寸等效为探测面强度与投影源分布的卷积

$$
I_{\rm blur} = I \otimes G\!\left(\sigma_{\rm eff}\right),\qquad \sigma_{\rm eff} = \sigma_{\rm src}\cdot \frac{z}{L}
$$

锥束模式下先按放大率 \(M = (L+z)/L\) 缩放坐标（探测面等效周期 \(pM\)、等效传播距离 \(z_{\rm eff} = z/M\)，即 Fresnel 缩放定理），再做同样的源模糊卷积。第一阶段建议只实现平行束 + 等效模糊，锥束作为接口预留。

**验收测试**：（1）单色、零源尺寸时退化为 B.5 结果；（2）源模糊 \(\sigma_{\rm eff}\) 等于半个条纹周期时，对应载频可见度下降到无模糊值的 \(\exp(-2\pi^2 \sigma_{\rm eff}^2 f_j^2)\) 附近（高斯模糊的解析衰减因子），偏差 \(<3\%\)。

### B.7 detector.py

三步：像素孔径积分（把高分辨强度按像素面元求和，等价于与 \(\mathrm{rect}\) 卷积后抽样，注意轴向与斜向载频受到的 \(\mathrm{sinc}\) 衰减不同——这是 \(a_j\) 探测面均衡的依据）；附加 MTF（高斯模糊，代表闪烁体/光学耦合）；噪声（按 `photons_per_pixel` 归一后加泊松噪声，输出多次实现用于统计可见度的不确定度）。要求像素周期与条纹周期不可公度性可配置，以便研究像素—条纹拍频。

### B.8 analysis.py

核心量的定义必须写死并文档化，避免歧义。

**傅里叶可见度**（主指标）：

$$
V_j = \frac{2\,|\hat I(\mathbf f_j)|}{\hat I(0)}
$$

由于载频严格落在 FFT 网格上且边界周期，直接取单点谱值即可；若加了噪声，取窗口内峰值。

**串扰矩阵**：

$$
C_{jk} = \frac{\int_{\Omega_j} |\hat I_k|^2\, d\mathbf f}{\int_{\Omega_j} |\hat I_j|^2\, d\mathbf f}
$$

其中 \(\hat I_k\) 通过只保留第 \(k\) 载频的参考解调获得；实现上以"窗口内非本载频能量占比"作为等效度量。

**互调审计**（对探测面强度谱重复 B.4 的 \(\rho_j\) 计算——传播和二值化都会产生新互调项，掩模面干净不代表探测面干净）。

**禁区距离检测**：对 \(V_j(z)\) 曲线找局部极小值与零点，输出每个载频的禁区清单及安全工作窗口（定义为所有 \(j\) 同时满足 \(V_j \geq 0.15\) 的 \(z\) 区间）。

### B.9 scans.py 与 report.py

三种扫描模式。`z_scan`：固定 \(E_0\)（可含带宽），扫 \(z\)，输出 \(V_j(z)\) 四条曲线 + 安全窗口标注（这就是多载频 Talbot carpet）。`energy_scan`：固定 \(z\)，扫单色能量，输出 \(V_j(E)\)，并叠加给定带宽下的谱加权可见度。`map_2d`：\(V_j(z,E)\) 二维图，四个载频各一张，外加 \(\min_j V_j(z,E)\) 综合图——**这张综合图就是最终交付的"能量—距离容忍度地图"**，直接标出 go 区域（\(\min_j V_j \geq 0.15\)）。

report.py 输出 JSON 摘要，字段包括：设计参数、碰撞清单、各扫描的安全窗口边界、目标工作点的 \(V_j\)、\(C_{jk}\)、\(\rho_j\)、噪声下可见度标准差，以及 go/no-go 布尔结论。全部原始数据存 HDF5。

---

## C. 容忍度分析流程（运行顺序）

按以下顺序执行，每步有明确通过标准，未通过则回到设计参数调整而不是继续往下跑。

**第 1 步，设计与碰撞检查**：`run_design.py` 生成 \(m=7, n=5\)、\(P=56\ \mu\mathrm m\)、Si、12.4 keV 的二值 π 光栅，碰撞检查 \(\rho_j < 0.02\) 全部通过。

**第 2 步，单色理想 z 扫描**：单色、零源尺寸、理想探测器，\(z \in [10, 300]\ \mathrm{mm}\)、120 步。确认存在 \(z^\ast\) 使四个 \(V_j\) 同时 \(> 0.3\)（理想条件下阈值取高）；记录各载频禁区。轴向载频（\(p=8\ \mu\mathrm m\)）与斜向载频（\(p\approx 7.92\ \mu\mathrm m\)）的 \(V(z)\) 曲线应几乎重合但有约 2% 的距离错位——验证这个错位在工作点处可忽略，这正是 \(m/n \approx \sqrt 2\) 公度设计的核心验证点。

**第 3 步，加入探测器**：像素 1 µm（同步辐射高分辨探测器场景）与 6.5 µm（sCMOS 直接场景，此时条纹欠采样，应观察到失败——这个负结果也要记录，它论证了实验室化必须走 MPG 包络路线）。通过标准：1 µm 像素下 \(\min_j V_j(z^\ast) \geq 0.2\)。

**第 4 步，能量容忍度**：固定 \(z^\ast\)，单色扫 10–16 keV，然后叠加 \(\Delta E/E \in \{1\%, 5\%, 10\%, 20\%\}\) 的高斯谱。交付 \(V_j\) 随带宽的退化曲线，并给出"可见度保持 \(\geq 0.15\) 的最大带宽"。物理预期：π 相移随 \(E\) 偏离而变为部分相移，且 Talbot 距离 \(\propto p^2 E\) 随能量漂移，两个效应叠加。

**第 5 步，源模糊容忍度**：扫 \(\sigma_{\rm eff}\) 从 0 到 2 µm，给出可见度—源尺寸曲线，反推给定 \(L\)、\(z\) 下允许的最大源尺寸。

**第 6 步，二维容忍度地图与 go/no-go**：输出 \(\min_j V_j(z,E)\) 地图（含真实探测器 MTF 与 \(10^4\) 光子/像素噪声），按 M1 里程碑标准判定：真实条件下每方向 \(V_j \geq 0.15\) 存在非空 \((z,E)\) 工作区，且工作区 z 向宽度 \(\geq \pm 5\%\)、能量向宽度 \(\geq \pm 3\%\)（否则实验对准与单色器稳定性要求过苛）。

**第 7 步（可选，进入优化）**：若第 6 步勉强通过或不通过，启动优化循环。推荐无梯度优化（CMA-ES）作用于低维参数 \((a_1..a_4, \psi_1..\psi_4, \tau, \epsilon)\)，目标函数取

$$
\mathcal L = -\min_j V_j(z^\ast) + \lambda_1 \max_j \rho_j + \lambda_2 \max_{j\neq k} C_{jk}
$$

先不做逐像素拓扑优化——那是第三阶段（Zernike/Fisher 逆设计）的事。

---

## D. 下一步计划：信号分析与重建（第二期开发规格）

第一期只回答"图样是否可用"；第二期回答"能否从图样恢复波前/相衬/dark-field"。以下是给代码 AI 的第二期任务分解。

### D.1 载频通道解调流水线

对带样品图 \(I_s\) 和参考图 \(I_r\)（或数值参考），对每个载频 \(j\)：傅里叶域以 guard band 半径开窗提取第 \(j\) 峰，移到原点，反变换得复包络 \(a_j^{s}(x,y)\)、\(a_j^{r}(x,y)\)。三个通道为

$$
\text{吸收：}\ \mu = -\ln\frac{|a_0^s|}{|a_0^r|},\qquad
\text{方向差分相位：}\ \Delta\psi_j = \arg\frac{a_j^s}{a_j^r},\qquad
\text{方向 dark-field：}\ D_j = -\ln\frac{|a_j^s|/|a_0^s|}{|a_j^r|/|a_0^r|}
$$

方向导数换算为 \(g_j = \mathbf n_j \cdot \nabla_\perp\varphi = \dfrac{\Delta\psi_j}{2\pi |\mathbf f_j|\, \lambda\, z_{\rm eff}} \cdot \dfrac{2\pi}{\lambda}\)（实现时统一写成折射角 \(\alpha_j = \Delta\psi_j\, /\, (2\pi |\mathbf f_j| z_{\rm eff})\)，再乘 \(k\) 得相位梯度，避免单位错误）。注意 π 相位光栅工作在半周期条纹时，条纹频率是载频的 2 倍，解调频率取实测强度谱主峰而非掩模载频——代码中以自动峰检测确定。

### D.2 多方向融合重建

**梯度重建**：加权最小二乘 \(\widehat{\nabla\varphi} = (A^T W A)^{-1} A^T W\, \mathbf g\)，\(W\) 逐像素取局部 \(V_j^2 \times\) 光子数（近似逆方差）。

**相位积分**：把 Kottler 复傅里叶积分法推广到四方向——构造两组正交对（\(0/90\) 与 \(45/135\) 旋转坐标系），各自复积分得 \(\varphi^{(1)}, \varphi^{(2)}\)，按残差加权融合；或直接解带 curl 正则的变分问题

$$
\min_\varphi \sum_j \int w_j \left| \mathbf n_j\cdot\nabla\varphi - g_j \right|^2 dx\,dy
$$

（傅里叶域一步闭式解，要求代码 AI 推导并实现闭式滤波器）。

**解包裹**：四方向冗余 + curl 一致性检测包裹点；变体 C（双环载频）的合成周期解包裹作为独立函数预留接口。

**各向异性 dark-field**：逐像素拟合 \(D(\theta_j) = D_0 + D_c\cos 2\theta_j + D_s\sin 2\theta_j\)，四方向恰好超定一个自由度，输出 \(D_0\)、各向异性度 \(\sqrt{D_c^2+D_s^2}/D_0\)、主方向 \(\theta_{\rm ani} = \frac{1}{2}\arctan(D_s/D_c)\) 三张图。

### D.3 蓝噪声通道（第二期后段）

窗口化零均值归一化互相关（ZNCC）追踪蓝噪声局部位移，窗口 \(16 \times 16\) 至 \(32\times32\) 像素、亚像素抛物插值；输出与载频通道独立的 \((\Delta x, \Delta y)\) 图。融合策略：载频通道给高空间频率、低噪声的梯度；蓝噪声通道给绝对参考（数值模拟参考图代替实验参考）与载频失效区补洞。UMPA 型代价函数版本作为可选升级。

### D.4 数字体模与验证指标

准备四个体模，全部解析定义、可算真值：（1）已知 Zernike 系数的纯相位波前（defocus + astigmatism + coma 组合），验证波前重建 RMSE 与斜率噪声 \(\sigma_\alpha\) 是否符合理论

$$
\sigma_\alpha \approx \frac{p_j}{2\pi z_{\rm eff} V_j \sqrt{N_\gamma}}
$$

（2）PMMA 球，验证 DPC 定量性与积分相位；（3）定向纤维束（一维高斯随机相位屏，相关长度各向异性），验证 \(\theta_{\rm ani}\) 估计精度目标 \(\pm 5^\circ\)；（4）含强梯度楔形，验证解包裹。

**关键 ablation**：同一体模、同等总剂量下对比 SB-BSTG 单发 vs checkerboard 单发 vs checkerboard 两次曝光（旋转 45°）。SB-BSTG 的论文级结论必须由这一组对比支撑：单发条件下各向异性 dark-field 参数精度不劣于两次曝光 checkerboard。

### D.5 交给 AI 工具的任务拆分建议

按依赖顺序分 8 个独立会话/任务提交，每个任务附带本文档对应小节 + 验收测试：任务 1 = B.1+B.2；任务 2 = B.3+B.4；任务 3 = B.5（要求先通过全部四项传播验收测试再继续）；任务 4 = B.6+B.7；任务 5 = B.8+B.9 + C 全流程脚本；任务 6 = D.1+D.2；任务 7 = D.3；任务 8 = D.4 体模与 ablation。给代码 AI 的提示中明确三点：所有物理公式以本文档为准、不得自行改约定；每个函数必须带单元测试且测试值来自解析结果（Talbot 距离、高斯模糊衰减因子、能量守恒）；先跑 \(1024^2\) 小网格验证正确性，再放大到生产网格。

一个容易被 AI 写错的点需要特别叮嘱：**可见度、互调审计、禁区距离都必须在"探测器面强度谱"上测，而不是在掩模透过率谱上测**；π 相位光栅探测面主条纹频率是掩模载频的 2 倍，解调窗口位置要跟随实测峰。

---

*Thinking...*

> I need to create a detailed specification in Chinese for a propagation kernel implementation with four acceptance tests that include concrete numerical reference answers—energy conservation, a fractional Talbot self-imaging test with a π-phase grating, an absorption grating at integer Talbot distance, and sampling convergence validation.
> 
> Now I'm working through the reference calculations: choosing a photon energy of 12.4 keV to get a wavelength near 0.1 nm, then computing the fractional Talbot distance for an 8 µm period π-phase grating, which comes out to 80 mm, where the intensity pattern should show period halving with high visibility for an ideal binary phase grating. So for a binary square wave with values 0 and 2 at 50% duty cycle, the Michelson visibility is exactly 1, but the Fourier visibility—defined as 2|Î(f)|/Î(0)—gives 4/π ≈ 1.27, which is the fundamental harmonic amplitude of the square wave. I need to be clear about which definition I'm using when reporting the visibility.
> 
> Now I'm checking whether the intensity at z_T/16 for an ideal π binary grating is truly binary 0/2 with full modulation. For a thin phase grating with ±1 transmission and 50% duty, the Fourier coefficients are nonzero only for odd harmonics, and after propagation through the Fresnel diffraction, the intensity pattern should show binary modulation with period p/2. Let me work through the exact computation to verify this matches the known literature result. I'm recalling the standard results from grating interferometry literature—a binary π-phase grating produces intensity patterns with period p/2 and maximal contrast at specific fractional Talbot distances, though the exact definition of the Talbot distance varies across references. Rather than getting bogged down in the definitions, I could simplify by looking at just the dominant diffraction orders: the ±1 orders with amplitudes around 2/π each, which through two-beam interference naturally create fringes with period p/2 regardless of propagation distance.
> 
> The higher-order terms (±3, ±5, etc.) are what introduce z-dependence and cause the contrast to wash out. At z = p²/(4λ), all odd harmonics acquire the same phase factor, leading to self-imaging and uniform intensity with zero visibility—this is the exact point where contrast vanishes. At z = p²/(2λ), the phase relationships shift again, and I need to work through how the harmonics combine to determine whether we get self-imaging or another specific pattern.
> 
> For the π-phase grating at z = p²/(8λ), I'm computing the intensity by summing the modified harmonics with their phase factors s_n. The classical Talbot array illuminator theory predicts this should produce a binary intensity pattern with period p/2 and unit visibility—essentially the self-image of an equivalent amplitude grating. Let me verify this by explicitly summing the first few harmonics to confirm the binary structure. Now I'm evaluating this series at x=0 to get a sum involving odd reciprocals with an alternating sign pattern based on residues modulo 8. I recognize this as a Dirichlet L-series with a character of period 8, where the character takes value +1 for n≡1,3 (mod 8) and −1 for n≡5,7 (mod 8), and I'm trying to verify whether this multiplicative character is consistent and determine its L-value at s=1. The series converges to π/(2√2) ≈ 1.1107, confirming the L-function value. This gives T̃(0) = √2, so the intensity at x=0 is 2, while at x=p/4 it drops to 0 due to the cosine terms vanishing for odd n. The pattern shows a binary intensity distribution with 50% duty cycle, yielding a Fourier visibility of 4/π ≈ 1.273 at the fundamental frequency.
> 
> For the numerical test, I'm setting up tight tolerances with N=4096 samples over 8p window width, giving 512 samples per grating period. The Fourier visibility should approach 4/π within 1% as oversampling increases, while energy conservation stays below 1e-10. The Michelson contrast bounds need to account for Gibbs oscillations near the binary edges, so I'll use Fourier visibility as the primary metric and evaluate intensity minima away from discontinuities to avoid edge artifacts.
> 
> For the second zero point at z₂ = p²/(4λ) = 160 mm, the field should equal e^{-iπ/4}T(x) exactly, making intensity uniform with std(I)/mean(I) < 1e-6. Under the discrete ASM, each harmonic n of the sampled square wave picks up phase exp(-iπn²/4), which should be constant across all odd harmonics within the Fresnel approximation. However, the exact ASM kernel includes higher-order corrections beyond Fresnel that scale with λ⁴f⁴ terms, which become relevant at high spatial frequencies near Nyquist.
> 
> I'm checking whether these higher-order phase errors matter: at the Nyquist frequency, the fourth-order correction is about 0.126 rad, but the square wave harmonics decay as 1/n, so high-frequency components have negligible amplitude. For moderate harmonics like n=15, the phase error drops to ~1.5e-6 rad, which is tiny. Even at n=101, the contribution remains negligible, so the Fresnel approximation should hold well enough for the uniformity test.
> 
> I'm setting up two tolerance thresholds to distinguish between kernels: with the Fresnel kernel, the self-imaging is exact in discrete arithmetic (std < 1e-10), while the full ASM kernel shows small deviations (std < 1e-3). This gives a nice diagnostic to validate the implementation.
> 
> For the absorption grating test, I'm using a binary amplitude pattern with 50% duty cycle and checking that the self-image correlation exceeds 0.999 at the Talbot distance. The Fresnel kernel should produce machine-precision identity, while the full ASM kernel will show small deviations.
> 
> I'm also adding a sub-test at half the Talbot distance where the self-image shifts by half a period, giving a contrast reversal that I can verify with correlation > 0.999.
> 
> For energy conservation with pure phase objects, I need to ensure all propagating modes are retained by disabling the band-limit window or setting it above Nyquist. Since the maximum normalized frequency here is 3.2e-3, well below 1, all modes propagate and conservation holds to machine precision.
> 
> I'm planning a sampling convergence test by comparing the visibility at different oversample rates—running the test 2 configuration at p/128 and p/512 samples per period to check that visibility changes by less than 0.5%.
> 
> Now I'm setting up the complete numerical reference values: using λ = 1.0 Å (E ≈ 12.398 keV), the Talbot distances z₁ = 80 mm, z₂ = 160 mm, z_T = 1.28 m, and z_T/2 = 0.64 m. The Fourier visibility reference is V = 4/π ≈ 1.27324 at the first grating harmonic, with intensity extrema at 0 and 2. At z₂ the field analytically equals e^{−iπ/4}T(x).
> 
> For the grid, I'm using a 2D propagator with the grating uniform in y, setting N_y = 64 to keep the 2D code path exercised. The window spans 8p = 64 µm with N_x = 4096, giving Δx = 15.625 nm = p/512, which ensures both the period and fundamental frequency land cleanly on the grid.
> 
> For convergence testing, I'm comparing N_x = 1024 (Δx = p/128) against N_x = 4096 and finding the Fourier visibility difference stays below 0.5%. The discrete square wave sampling introduces errors that scale as O((n/N_p)²) for harmonic n, so with N_p = 128 samples per period the correction is negligible. To avoid sign ambiguity at exact zero crossings, I'm sampling at cell centers x_i = (i+0.5)Δx or using a floor-based definition like T_i = +1 if frac(x_i/p) < 0.5 else −1.
> 
> For the reference values, the DC coefficient Î(0) = 1 by energy conservation, and the fundamental at f = 2/p has magnitude |c| = 2/π, giving a visibility ratio |Î(f=2/p)|/Î(0) ≈ 0.636. I'm also deriving how visibility varies with propagation distance z by tracking the intensity harmonic at 2/p as a sum over field harmonic pairs, which oscillates with period p²/(2λ) in z. I'm working through the pairing structure of these terms—the imaginary components from n=1 and n=−3 cancel out, leaving real contributions. The n=−1 term gives 4/π², while subsequent pairs like (3,−5) and (7,−9) contribute additional real terms with a clear pattern emerging in the series.
> 
> Now I'm summing up the coefficients: the series becomes 1 + 2(1/15 − 1/63 + 1/143 − 1/255 + ...), where the denominators are products of consecutive odd numbers. Computing the partial sums numerically, I'm tracking how the alternating series converges toward the final amplitude value.
> 
> The result is remarkably clean: A₂ works out to give V(40mm) = 2√2/π ≈ 0.9003, and checking this against a sinusoidal hypothesis V(z) = (4/π)|sin(πz/160mm)| matches perfectly. The voltage profile is sinusoidal with zeros at the boundaries and a peak at the midpoint.
> 
> Refining the general form, I'm testing V(z) = (4/π)|sin(4πλz/p²)| — at z=80mm this gives sin(π/2)=1, at z=40mm it yields sin(π/4)=√2/2, and at z=160mm it returns sin(π)=0, all matching the computed values. This appears to be the analytic reference curve for an ideal binary π grating with 50% duty cycle under monochromatic illumination, capturing the Fourier visibility at the second harmonic frequency. The pairwise harmonic cancellations I found earlier seem to have conspired to produce this pure sinusoidal form. Now I'm working through the Fourier series expansion to evaluate those sums over odd integers. Using the known result that the sum of sin(nu)/n over positive odd n equals π/4 in the range 0 < u < π, I can substitute back and simplify the expression for A₂. After careful algebra with the exponential terms and imaginary unit manipulations, this yields the exact result: the visibility follows V = (4/π)|sin(4πλz/p²)|, which confirms the sine dependence rigorously.
> 
> The domain constraint is 0 < z < p²/(4λ), corresponding to 160 mm, with periodicity beyond that. I'm also noting that the intensity pattern at the binary grating (0/2 values) aligns perfectly with the visibility formula since a binary square wave has a fundamental Fourier component of 4/π. For the absorption grating case with 50% duty cycle, I'm working through the Fourier coefficients where only odd harmonics appear, and examining how the visibility at the fundamental frequency f = 1/p depends on z through cross-terms between the zero and first-order components. The visibility formula V₁(z) = (4/π)|cos(πλz/p²)| has zeros at z = 0.32 m and 0.96 m, with maxima at z = 0, 0.64 m, and the self-imaging distance. For the second harmonic component A₂, I need to check pairs separated by 2 in the Fourier series—like (1,−1) and (3,1)—which contribute constant and oscillating terms respectively, though these higher harmonics don't affect the main visibility test. The numerical results should match the analytic formulas closely since the Fresnel approximation is valid at these low spatial frequencies.
> 
> For the test tolerances, the phase error from higher-order Fresnel terms (λ⁴f⁴/8) is completely negligible at n=1,2, so a 0.2% tolerance is appropriate. I'm setting up the energy conservation test with a 1024×1024 random phase screen or binary grating mask, propagating to z=80 mm and checking that the total intensity remains constant to within floating-point roundoff error (~1e−12). For the convergence test, I'm comparing the visibility V(z₁) across different sampling rates—64, 128, and 512 samples per period—to verify that the discrete Fourier coefficients converge to the analytic 4/π value, with the deviation dropping below 0.2% at 128 samples.
> 
> The discrete square wave's DFT coefficient deviates from the continuous 2/π by roughly (π/N_p)²/6, which at N_p=64 gives only ~4e−4 relative error—negligible compared to visibility effects from higher harmonics near Nyquist. I should also document the propagator API specification, W_BL details, complex128 dtype requirements, and fftshift conventions, plus warn about critical pitfalls: the sign convention of the propagation kernel must match the forward e^{+ikz} with the time convention e^{−iωt}, and the band-limit window should not be applied during these periodic tests.
> 
> For the deliverable, I'm structuring a single self-contained prompt in Chinese that includes the role, physics context, API spec, formulas, test specifications with numeric tolerances, and a pitfalls section. The intensity validation at z₁ will use robust criteria: V(z₁) = 4/π ± 0.5%, median of the top 40% of pixels between 1.95 and 2.05, and median of the bottom 40% between 0 and 0.05. I'll also add an analytic curve test sampling 25 z values from 5 to 315 mm against the theoretical (4/π)|sin(4πλz/p²)| with a maximum absolute error tolerance of 0.01 in V units, and an absorption test using normalized cross-correlation at z_T. Now I'm adding the visibility curve for the absorption grating using the formula V₁(z) = (4/π)|cos(πλz/p²)|, calculating values at a few key distances like 0.16 m to show how visibility evolves, and I'll include a small reference table. Then I'm structuring the full prompt with sections covering the task description, required API functions with kernel options and band-limiting, physical formulas and conventions, global test parameters using the 2D propagator with N_y=16, and detailed specs for both tests including the visibility formula V(z)=(4/π)|sin(4πλz/p²)| with precomputed values at standard distances. I need to pin down the wavelength directly as 1.0×10⁻¹⁰ m to sidestep conversion errors, and I'm setting up all tests with plane waves at unit amplitude and normalized intensity. I'm also sketching out the key derivations—like why odd n satisfies n²≡1 mod 8, and how the Fresnel field at z₂ becomes e^{−iπ/4}T(x) exactly—so the reference values are trustworthy. Now I'm laying out the tolerance thresholds: energy error under 1e−10 for test 1, visibility within 0.5% at the self-image point for test 2a, and for test 3 the normalized cross-correlation needs to stay above 0.999 with intensity differences capped at 1e−8 for Fresnel propagation.
> 
> I'm also defining the API with propagate and fourier_visibility functions, setting up grating generators for both phase and amplitude masks, and clarifying that bandlimit stays off during tests while evanescent waves get zeroed out when λ²f²>1.
> 
> Now I'm computing the exact numerical constants: 4/π ≈ 1.27324, 2√2/π ≈ 0.90032, and the sinusoidal coefficients for the visibility formula at different phase shifts. I'm also mapping the propagation distances to dimensionless parameters—at z=0.16 m the phase parameter u reaches π, and at z=0.08 m it's π/2—then verifying the absorption parameter v aligns with the expected zero crossing at z=0.32 m.
> 
> For the computational grid, I'm setting the window to 64 µm with 4096 points giving 15.625 nm spacing and 512 samples per wavelength period. The frequency grid spacing works out so that the fundamental frequency is exactly 8 grid steps and the second harmonic is 16 steps. I'm checking that all frequencies stay below the Nyquist limit to ensure purely propagating modes.
> 
> The test specifications call for a 1024² 2D array with a random binary phase screen at z=80 mm for the first test, and for convergence testing I'm varying the grid resolution from 512 to 4096 points to verify the visibility measurements converge to within 0.5% relative error. The Fresnel propagation kernel...
> 
> Defining the propagation kernel... The zero-frequency component of the intensity spectrum equals the mean intensity—for phase gratings this is 1 (due to conservation), while for absorption gratings it's 0.5. I'll structure the final report as JSON with fields for each test including name, measured value, expected value, tolerance, and pass/fail status, and note that runtime should be just seconds on a standard laptop without GPU acceleration. For the derivation appendix, I'm documenting the key mathematical facts: odd harmonics at the specific propagation distance acquire a shared phase factor that produces uniform intensity; the second-order visibility follows from Fourier analysis of the grating pattern; and absorption gratings have a simpler two-term structure leading to a different visibility formula.

# 任务 3 提示词：X 射线角谱传播核 + 四项验收测试（含数值参考答案）

下面横线之间的全部内容就是可直接粘贴给代码生成 AI 的完整提示词。所有参考数值我已做过解析推导（推导概要附在提示词末尾的附录中，便于你和代码 AI 审计），不是拍脑袋给的经验值。

---

## 角色与任务

你是一名科学计算工程师。请用 Python 3.11 + numpy/scipy 实现一个 X 射线单色相干传播模块 `propagate.py` 及其完整测试套件 `test_propagate.py`。本模块是一条 X 射线光栅干涉仪模拟流水线的核心，正确性要求极高：**四项验收测试全部通过后任务才算完成**，每项测试都有解析参考答案和明确容差，不允许放宽容差来"凑通过"。

## 一、全局约定

单位统一为米。二维数组 axis 0 为 y、axis 1 为 x。所有场用 `complex128`，禁止 float32。时间相位约定 \(e^{i(kz-\omega t)}\)，前向传播核取正指数。FFT 用 `numpy.fft`，频率轴由 `fftfreq(N, d=dx)` 生成，**传播核直接构建在未 shift 的频率网格上**，避免 shift 不一致的错误。

所有测试固定波长 \(\lambda = 1.0000\times10^{-10}\ \mathrm{m}\)（即 1 Å，对应约 12.398 keV；直接以波长为输入，不做能量换算，避免常数不一致）。入射为单位振幅平面波，入射面平均强度归一为 1。

## 二、需实现的 API

```python
def propagate(U0: np.ndarray, dx: float, wavelength: float, z: float,
              kernel: str = "asm", bandlimit: bool = False) -> np.ndarray:
    """二维角谱传播。U0: 复场 (Ny, Nx)；dx: 采样间隔 [m]；z: 传播距离 [m]。
    kernel="asm":    H = exp(i*2πz/λ * sqrt(1 - λ²(fx²+fy²)))，λ²f²>1 处 H=0（截断倏逝波）
    kernel="fresnel":H = exp(-i*π*λ*z*(fx²+fy²))   （常数相位 e^{ikz} 可省略，测试只看强度）
    bandlimit=True 时按 Matsushima & Shimobaba 带限准则加窗；本测试套件全部取 False。
    """

def make_pi_grating(Ny: int, Nx: int, dx: float, p: float,
                    phase: float = np.pi) -> np.ndarray:
    """一维二值相位光栅沿 x，y 方向均匀。x_i = (i - Nx/2)*dx；
    (x_i mod p) < p/2 处透过率 exp(i*phase)，其余为 1。返回复透过率 (Ny, Nx)。"""

def make_amp_grating(Ny: int, Nx: int, dx: float, p: float) -> np.ndarray:
    """同上几何，但 (x mod p) < p/2 处透过率 1，其余为 0（二值吸收光栅）。"""

def fourier_visibility(I: np.ndarray, dx: float, f_target: float) -> float:
    """强度图先沿 y 平均得一维剖面，再 FFT。
    返回 V = 2*|Î(f_target)| / |Î(0)|。要求 f_target 严格落在频率网格上，
    否则 raise（用 np.isclose 判断，容差 1e-6 倍频率步长）。"""
```

## 三、测试公共参数

除测试 1 外，统一使用：光栅周期 \(p = 8\ \mu\mathrm{m}\)；窗口宽度 \(L_x = 8p = 64\ \mu\mathrm{m}\)（保证 \(f = 1/p\) 与 \(2/p\) 严格落在 FFT 网格上，分别是第 8 与第 16 个频率格点）；\(N_x = 4096\)（每周期 512 采样点），\(N_y = 8\)（y 向均匀，仅为了走通二维代码路径）；周期边界条件天然满足，**不加任何窗函数，不做零填充**。

派生参考量（代码中 assert 这些值以确认几何正确）：\(p^2/\lambda = 0.64\ \mathrm{m}\)；π 光栅最大对比距离 \(z_1 = p^2/(8\lambda) = 0.080\ \mathrm{m}\)；π 光栅零对比距离 \(z_2 = p^2/(4\lambda) = 0.160\ \mathrm{m}\)；吸收光栅 Talbot 距离 \(z_T = 2p^2/\lambda = 1.28\ \mathrm{m}\)。

构造光栅后必须 assert：二值占空比与 0.5 的偏差 \(< 10^{-12}\)（按采样点计数）。

## 四、测试 1：能量守恒

网格 \(1024\times1024\)，\(dx = 15.625\ \mathrm{nm}\)。物体为纯相位随机屏：每像素独立取相位 0 或 π（固定随机种子 42），\(U_0 = e^{i\varphi}\)。用 `kernel="asm"`、`bandlimit=False` 传播 \(z = 0.080\ \mathrm{m}\)。

**通过标准**：\(\left|\sum|U_z|^2 - \sum|U_0|^2\right| / \sum|U_0|^2 \leq 10^{-10}\)。

说明：此网格上最大 \(\lambda f \approx 3.2\times10^{-3} \ll 1\)，无倏逝分量，酉传播核下守恒应达浮点精度量级。若误差在 \(10^{-3}\) 量级，几乎必然是核公式或频率网格错了，不是浮点问题。

## 五、测试 2：π 相位光栅分数 Talbot 效应（核心锚点测试）

使用 `make_pi_grating`，ASM 核，测量强度在 \(f = 2/p = 2.5\times10^{5}\ \mathrm{m^{-1}}\) 处的傅里叶可见度（注意：**π 光栅的主条纹频率是光栅频率的 2 倍**，在 \(1/p\) 处测是常见错误，那里的可见度接近零）。

**2a. 峰值距离检验**。在 \(z_1 = 80\ \mathrm{mm}\)：理想解析结果是强度为周期 \(p/2\)、取值 \(\{0, 2\}\)、占空比 50% 的二值图样，其傅里叶可见度为

$$
V(z_1) = \frac{4}{\pi} = 1.273240
$$

通过标准：\(|V_{\rm num} - 1.273240| \leq 0.006\)。辅助检查：将强度像素排序，最亮 40% 像素的中位数落在 \([1.95, 2.05]\)，最暗 40% 像素的中位数 \(\leq 0.05\)，且全图平均强度与 1 的偏差 \(\leq 10^{-10}\)（相位光栅不吸收）。

**2b. 零点距离检验**。在 \(z_2 = 160\ \mathrm{mm}\)：解析上场恰好复原为 \(e^{-i\pi/4}\,T(x)\)，强度严格均匀。通过标准：\(V_{\rm num}(z_2) \leq 0.005\)；且 \(\mathrm{std}(I)/\mathrm{mean}(I) \leq 10^{-3}\)（ASM 核）。**换用 Fresnel 核重跑此点，要求 \(\mathrm{std}(I)/\mathrm{mean}(I) \leq 10^{-9}\)**——Fresnel 核下离散自成像是精确的，此项能把两种核的实现差异钉死。

**2c. 全曲线检验**。解析可见度曲线为

$$
V(z) = \frac{4}{\pi}\left|\sin\!\left(\frac{4\pi\lambda z}{p^2}\right)\right|
$$

在 \(z \in \{20, 40, 60, 80, 100, 120, 140, 160, 200, 240\}\ \mathrm{mm}\) 十个点上逐点比对，参考值表：

| z (mm) | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 200 | 240 |
|---|---|---|---|---|---|---|---|---|---|---|
| V 解析 | 0.48724 | 0.90032 | 1.17632 | 1.27324 | 1.17632 | 0.90032 | 0.48724 | 0.00000 | 0.90032 | 1.27324 |

通过标准：每点 \(|V_{\rm num} - V_{\rm ana}| \leq 0.01\)（绝对误差，含零点）。

## 六、测试 3：吸收光栅 Talbot 自成像

使用 `make_amp_grating`，ASM 核。记 \(I_0 = |T|^2\)（即 z=0 强度，二值 0/1）。

**3a. 整数 Talbot 距离**。传播到 \(z_T = 1.28\ \mathrm{m}\)，计算 \(I(z_T)\) 与 \(I_0\) 的 Pearson 相关系数（y 平均后的一维剖面）。通过标准：\(\mathrm{NCC} \geq 0.999\)（ASM 核）；换 Fresnel 核后 \(\max|I(z_T) - I_0| \leq 10^{-8}\)（离散精确自成像）。

**3b. 半 Talbot 距离反衬**。传播到 \(z_T/2 = 0.64\ \mathrm{m}\)，图样应等于 \(I_0\) 平移 \(p/2\)（即 256 个采样点，用 `np.roll` 构造参考）。通过标准：\(\mathrm{NCC}\big(I(z_T/2),\ \mathrm{roll}(I_0, 256)\big) \geq 0.999\)。

**3c. 可见度曲线**。吸收光栅强度在 \(f = 1/p\) 处的可见度有精确闭式解（只有两项谱对贡献，见附录）：

$$
V_1(z) = \frac{4}{\pi}\left|\cos\!\left(\frac{\pi\lambda z}{p^2}\right)\right|
$$

参考值：\(V_1(0.08\ \mathrm m) = 1.17632\)，\(V_1(0.16) = 0.90032\)，\(V_1(0.32) = 0\)，\(V_1(0.48) = 0.90032\)，\(V_1(0.64) = 1.27324\)。逐点通过标准同 2c（\(\leq 0.01\)）。注意此光栅平均强度为 0.5，`fourier_visibility` 的归一化自动处理这一点。

## 七、测试 4：采样收敛性

重复测试 2a（\(z_1 = 80\ \mathrm{mm}\)），但 \(N_x \in \{512, 1024, 4096\}\)（每周期 64、128、512 采样点，窗口尺寸不变）。通过标准：三个分辨率下 \(|V - 4/\pi|\) 单调不增；且 \(|V_{N=1024} - V_{N=4096}| / V_{N=4096} \leq 0.5\%\)。

## 八、已知陷阱清单（实现前通读）

第一，可见度在**强度谱**上测，不在透过率谱上测。第二，π 光栅解调频率是 \(2/p\)。第三，窗口必须是 \(p\) 的整数倍，`fourier_visibility` 必须校验 f_target 落在网格上并在不满足时抛异常，而不是静默取最近格点。第四，光栅占空比必须按采样点严格等于 50%，构造后 assert。第五，传播核在未 shift 的 `fftfreq` 网格上直接构建。第六，全部测试 `bandlimit=False`；带限窗照常实现但不在本套件中启用。第七，复现性：随机屏固定种子。第八，别用 float32，别用 `rfft` 后忘记频率轴对应关系。

## 九、输出格式

测试脚本运行后打印一张 pass/fail 表（测试项、实测值、参考值、容差、结论），并把同样内容写入 `results/test_propagate_report.json`。任何一项 fail 时进程退出码非零。预期总运行时间在普通笔记本上不超过 30 秒，无需 GPU。

## 附录：参考答案的推导概要（供审计，不需实现）

记光栅复透过率的傅里叶系数为 \(c_n\)（谐波频率 \(n/p\)），Fresnel 传播给第 \(n\) 阶附加相位 \(-\pi\lambda z n^2/p^2\)。

（1）π 光栅 \(T = \pm1\) 只含奇次谐波，而奇数 \(n\) 满足 \(n^2 \equiv 1 \pmod 8\)。故在 \(z_2 = p^2/(4\lambda)\) 处相位因子 \(e^{-i\pi n^2/4} = e^{-i\pi/4}\) 对所有奇 \(n\) 相同，场整体复原，强度严格均匀——这是 2b 的依据。

（2）强度在 \(f = 2/p\) 的系数为 \(A_2(z) = \sum_{n\ \rm odd} c_n c_{n+2}\, e^{-i(n+1)u}\)，\(u = 4\pi\lambda z/p^2\)。代入 \(c_n c_{n+2} = -4/(\pi^2 n(n+2))\)，做部分分式 \(1/(n(n+2)) = \tfrac12[1/n - 1/(n+2)]\)，并利用 \(\sum_{n\ \rm odd>0}\sin(nu)/n = \pi/4\)（\(0<u<\pi\)），得精确结果 \(A_2 = (2/\pi)\sin u\)，故 \(V(z) = (4/\pi)|\sin(4\pi\lambda z/p^2)|\)——这是 2c 全表的来源，2a 是其峰值 \(4/\pi\)。

（3）吸收光栅 \(c_0 = 1/2\)，偶次谐波（除 0 外）为零，故强度在 \(f=1/p\) 的系数只有 \((c_1 c_0)\) 与 \((c_0 c_{-1})\) 两项：\(A_1 = (1/\pi)\cos(\pi\lambda z/p^2)\)，平均强度 \(\sum|c_n|^2 = 1/2\)，故 \(V_1(z) = (4/\pi)|\cos(\pi\lambda z/p^2)|\)——这是 3c 的来源；\(z_T\) 与 \(z_T/2\) 的自成像/半周期平移由 \(e^{-i\pi n^2 z/(p^2/\lambda)}\) 在整数与半整数 Talbot 距离的取值直接得出。

（4）二值 0/2、占空比 50% 的方波，基频复系数模为 \(2/\pi\)，按 \(V = 2|\hat I(f)|/\hat I(0)\) 定义得 \(4/\pi \approx 1.2732\)。注意傅里叶可见度对方波可以大于 1，这不是 bug；对应的 Michelson 可见度为 1。

---

两点使用建议。第一，把这份提示词整体交给代码 AI 后，若它第一轮有测试不过，优先怀疑的顺序是：解调频率取错（\(1/p\) vs \(2/p\)）、核建在 shift 过的频率网格上、占空比未对齐采样点——这三个错误覆盖了此类实现约九成的失败案例。第二，测试 2b 和 3a 的"Fresnel 核达机器精度、ASM 核有约 \(10^{-3}\) 以下小偏差"这对双重标准非常有诊断价值：如果两种核给出完全相同的结果，说明 ASM 核可能被错误地实现成了 Fresnel 展开，值得专门核查。

通过全部四项后，这个传播核就可以作为可信基座接入任务 4（多色 + 部分相干 + 探测器），那一层的验收测试（高斯模糊解析衰减因子）我可以在你需要时同样展开成带数值参考的提示词。

---

# Next optimization and validation plan (2026-07-13)

This section supersedes any earlier instruction that treats the current binary thickness map or
matched cosine-carrier reconstruction as fabrication-ready. The current result is a useful numerical
prototype, but the next phase must optimize the complete physical imaging system and validate it with
an independent forward model.

## 1. Final research objective

The target is a manufacturable single-shot X-ray phase grating that recovers phase, transmission,
mean dark-field, dark-field anisotropy, and principal scattering direction at equal or lower total
dose than grid, checkerboard, and established omnidirectional gratings.

The project succeeds only if one candidate satisfies all four requirements:

1. **Physical correctness:** material-dependent phase and absorption, partial coherence, spectrum,
   detector sampling, MTF, and noise are included.
2. **Identifiability:** the directional-dark-field design matrix has rank 3 and condition number at
   most 2.0 throughout the usable field.
3. **Manufacturability:** the exported layout passes line/space, corner, aspect-ratio, connectivity,
   depth, and tiling rules without more than 5% mask repair.
4. **Measured superiority:** at matched total dose and matched effective spatial resolution, it beats
   the strongest baseline by at least 15% in the preregistered reconstruction score or reaches the
   same error with at least 20% fewer photons.

Do not use mean carrier visibility alone as the optimization objective. Lower visibility may be
acceptable if the pattern provides substantially more independent directional information, but the
weakest carrier must remain above the detector-specific reliability threshold.

## 2. Immediate corrections before further optimization (Milestone M0)

### 2.1 Implement material-dependent grating transmission

Replace the current `h_pi = wavelength / 2` implementation with

\[
h_\pi(E,m)=\frac{\lambda(E)}{2\delta_m(E)}, \qquad
T_m(x,y,E)=\exp[-k\beta_m(E)t(x,y)]\exp[-ik\delta_m(E)t(x,y)].
\]

Use xraylib as the authoritative runtime source for `delta` and `beta`. Store the material, density,
design energy, substrate, thickness, and optical constants in every report. At 12.398 keV, the first
reference test should reproduce approximately 15.79 um for a silicon pi phase shift and approximately
0.936 intensity transmission through that thickness.

**M0 acceptance tests:**

- xraylib and independently tabulated `delta/beta` agree within 1%;
- phase delay through `h_pi` is pi within 0.5%;
- simulated absorption agrees with the Beer-Lambert expression within 0.5%;
- the saved thickness map is in micrometres and the report never labels a 0.05 nm map as physical.

### 2.2 Separate the forward model from reconstruction

The forward simulation must generate the sample image from wave propagation through an independent
sample and physical grating. Reconstruction may receive only the noisy reference/sample images,
calibration parameters, and nominal carrier locations. It must not receive truth maps or ideal
carrier envelopes.

For each pattern, use its actual propagated per-carrier visibility. Do not assign the same scalar
visibility to star, grid, checkerboard, circular, or hexagonal candidates.

Add these modules or equivalent ownership boundaries:

```text
src/materials.py             delta/beta, pi height, substrate absorption
src/detector.py              pixel aperture, binning, MTF, gain, Poisson/read noise
src/physical_forward.py      source -> sample -> grating -> detector propagation
src/fabrication.py           DRC, morphology, sidewall/depth/CD perturbations
src/candidate_library.py     reproducible pattern families and parameters
src/system_optimization.py   Pareto search using physical reconstruction metrics
scripts/validate_end_to_end.py
scripts/export_fabrication_layout.py
```

## 3. Candidate grating families

Every candidate must be evaluated using the same material, illuminated area, total dose, detector,
effective image resolution, and number of reference images. Previously published concepts are
included as controls, not presented as novel inventions.

| ID | Structure | Role | Expected advantage | Main risk | Novelty position |
|---|---|---|---|---|---|
| S4 | Current four-carrier star `(7,0),(0,7),(5,5),(5,-5)` | corrected baseline | balanced four-direction fit and good phase conditioning | modulation divided among four carriers; small tips and gaps | exact carrier/layout may be distinctive, but single-shot 2-D D is prior art |
| H3 | Three-carrier hexagonal phase grating at 0, 60, 120 degrees | **primary new candidate** | rank-3 D fit with 25% fewer carriers; potentially higher minimum visibility | carrier collisions after binarization; hexagonal patterns have prior art | test novelty of this physical carrier implementation and optimization, not hexagonal sensitivity itself |
| H3-4L | Four-level version of H3 | high-efficiency candidate | improved diffraction efficiency and carrier balance | two or three etch depths and tighter depth control | multilevel implementation is established; system-level performance may be publishable |
| DRH6 | Dual-radius hexagonal pattern: three directions at two carrier magnitudes | high-risk research candidate | two directional correlation-length channels in one exposure; possible multi-scale D | six carriers reduce visibility and increase crosstalk | potentially strongest pattern-level novelty if single-shot multi-scale D is demonstrated |
| FATO | Fabrication-aware topology-optimized rounded-cell grating | optimization candidate | directly maximizes Fisher information under line/space and corner constraints | optimization complexity and local minima | novelty rests on objective and experimentally verified topology |
| CUC | Circular-unit-cell grating with square and hexagonal packing | mandatory literature baseline | established omnidirectional single-shot sensitivity | may require high-resolution detector and bright source | prior art; never claim as new |
| RBN | Random or blue-noise phase/absorption modulator | broadband robustness baseline | less Talbot tuning, unique local features, anisotropic blur-tensor retrieval | lower spatial resolution and more difficult calibration | speckle/random-mask imaging is prior art |
| MPG | Two-dimensional modulated phase grating | detector-resolution baseline | directly resolvable fringes with microfocus source | extra envelope parameters and fabrication complexity | active prior art; use as comparator where detector pixels are large |
| G2/C2 | grid and checkerboard, one and two shots | minimum controls | high raw visibility and simple fabrication | rank-deficient D in one shot or dose split in two shots | established baseline |

### 3.1 Primary novelty hypothesis: H3

For three equally spaced sensitivity angles, use

\[
\theta_j \in \{0,\pi/3,2\pi/3\}, \quad
A_j=[1,\cos(2\theta_j),\sin(2\theta_j)].
\]

This matrix has rank 3 and good conditioning. The hypothesis is that H3 retains D0/Dc/Ds
identifiability and two-dimensional phase-gradient recovery while concentrating the modulation budget
into three carriers rather than four. Optimize integer carrier pairs that approximate equal magnitude
and 60-degree spacing on the periodic FFT lattice. Candidate supercells may use triangular/hexagonal
sampling internally, but the exported layout and detector simulation must preserve real Cartesian
coordinates.

**H3 superiority test:** H3 must improve minimum visibility by at least 15% or reduce directional-D
angle variance by at least 20% relative to S4 at matched dose and resolution, without increasing
phase NRMSE or contamination by more than 5%.

### 3.2 High-risk novelty hypothesis: DRH6

Use two carrier radii `f1` and `f2` at the same three angles. The two radii probe different effective
dark-field correlation lengths and may allow single-shot directional plus scale-sensitive scattering.
The reconstruction must explicitly fit separate visibility-loss parameters at the two radii. This
candidate is pursued only if the single-radius H3 passes M3; otherwise it is too likely to spend the
visibility budget on redundant channels.

**DRH6 superiority test:** recover two simulated scatterer-size populations with less than 15%
relative error and direction MAE below 7 degrees at 10,000 photons/pixel, while all six carriers retain
visibility at least 0.12 after detector MTF and spectrum integration.

## 4. Fabrication-aware pattern generation (Milestone M1)

Enforce geometry before propagation, not as a cosmetic postprocessing step. Initial silicon coupon
rules are:

| rule | initial value |
|---|---:|
| minimum printed line/space | 2.0 um |
| minimum isolated feature diameter | 2.0 um |
| minimum corner radius | 0.5 um |
| maximum repaired area fraction | 5% |
| nominal Si depth at 12.4 keV | about 15.8 um |
| maximum local aspect ratio target | 10:1 |
| depth tolerance simulated | +/-10% |
| critical-dimension bias simulated | +/-0.5 um |
| sidewall angle simulated | 85-90 degrees |

Implement opening/closing, connected-component removal, minimum-neck checks, corner rounding, and
periodic-boundary checks. Recompute all Fourier metrics after every repair. Reject rather than repair
a candidate if the repaired area exceeds 5% or a target carrier falls by more than 10%.

Export a tiled vector layout (prefer GDSII) plus a raster audit image. The exported layout must include
alignment marks, design ID, material/depth metadata, and star/grid/checkerboard/H3 controls on the same
coupon.

## 5. Optimization formulation (Milestones M2-M3)

### 5.1 Variables

Optimize carrier integer vectors, amplitudes, phases, fill factor, phase levels, supercell size,
material/depth, propagation distance, and detector sampling. For topology candidates, optimize a
signed-distance or density field followed by a differentiable or projected fabrication filter.

### 5.2 Physical objective vector

Use Pareto optimization; do not collapse the search to one arbitrary scalar until the Pareto front is
available. Minimize:

```text
phase_NRMSE
transmission_RMSE
D0_relative_RMSE
D_anisotropy_RMSE
D_angle_MAE_deg
worst_case_CRLB_trace
carrier_balance_CV
carrier_contamination
fabrication_penalty
energy_distance_failure_fraction
```

Maximize:

```text
minimum_pattern_specific_visibility
smallest_Fisher_information_eigenvalue
coverage_area_in_(energy,distance)
information_per_photon
information_per_carrier
```

Report the full Pareto front. Select a fabrication candidate only after applying the fixed gates in
Section 9.

### 5.3 Search sequence

1. Coarse enumerate symmetry families, integer carrier sets, phase levels, fill factors, and distances.
2. Run multi-start Bayesian or evolutionary search over continuous amplitudes/phases and system
   settings.
3. Refine the best 10 candidates with local derivative-free or level-set optimization.
4. Apply DRC and re-evaluate; never rank a pre-DRC mask as the winner.
5. Re-run the final five candidates with independent seeds, dense tolerances, and the full detector
   model.

## 6. Physical forward model (Milestone M2)

The production simulation must include:

- incoherent energy integration over a measured or specified spectrum;
- energy-dependent material phase and absorption;
- finite source size and cone-beam magnification;
- partial spatial coherence;
- substrate transmission and optional substrate phase curvature;
- detector pixel-aperture integration, scintillator/optical MTF, gain, Poisson noise, and read noise;
- reference/sample flat-field drift and small grating translations/rotations;
- fabrication depth, CD, sidewall, roughness, stitching, missing-feature, and bow perturbations.

Dark-field must be simulated independently of the reconstruction model. Implement at least two of:

1. convolution with a direction-dependent small-angle scattering kernel;
2. anisotropic Fokker-Planck diffusion of the structured illumination;
3. Monte Carlo angular scattering for selected validation cases.

Agreement between two independent dark-field forward models should be within 5% for the simple
Gaussian-scattering phantom before optimization results are trusted.

## 7. Validation phantoms and reconstruction (Milestone M3)

Use a training/validation split. Optimization may see the training phantoms; final ranking uses frozen
validation phantoms.

Required validation set:

| channel | phantoms |
|---|---|
| phase | low-frequency periodic, Zernike modes, sharp aperture, strong gradient, phase-wrap case |
| T | smooth attenuation, sharp edges, weak/strong absorption, beam-hardening case |
| D0 | uniform, smooth spatial variation, sharp inclusion, multiple strengths |
| directional D | fibres at 0/15/30/45/60/75/90 degrees, crossed fibres, spatially varying angle |
| model mismatch | non-Gaussian scattering, two orientations per pixel, attenuation/phase edges |

Reconstruction outputs must include uncertainty maps. Compare empirical errors with the Fisher/CRLB
prediction, and require empirical variance no more than 1.5 times the predicted standard deviation in
well-modelled regions. Report bias separately from variance.

All method comparisons must use:

- equal incident total photons, including multiple exposures;
- each pattern's measured/simulated carrier visibility;
- the same object and detector;
- matched effective spatial resolution, measured from point/edge response rather than smoothing sigma;
- at least 30 noise realizations for development and 100 for the final table;
- bootstrap 95% confidence intervals for all headline differences.

## 8. Dense tolerance and robustness study (Milestone M4)

Around every candidate's nominal operating point, use at least:

```text
energy: nominal +/-10%, spacing <=0.25% of nominal energy
distance: nominal +/-10%, spacing <=0.5% of nominal distance
source FWHM: 0 to the maximum planned experimental value
detector MTF sigma: nominal +/-25%
depth: nominal +/-10%
CD bias: -0.5 to +0.5 um
sidewall angle: 85, 87.5, 90 degrees
rotation: +/-0.5 degrees
translation: one full detector fringe period
flat-field gain drift: 0-2%
```

After the grid sweep, run at least 500 joint Monte Carlo tolerance samples using realistic correlated
fabrication errors. Save failure maps identifying which carrier or reconstruction channel violates a
gate.

## 9. Fixed go/no-go gates

These gates must be declared before viewing final validation results.

### M0 physics gate

- material phase/absorption tests pass;
- independent propagation tests pass;
- no idealized `lambda/2` thickness remains in fabrication output.

### M1 fabrication gate

- all line/space/corner/aspect-ratio rules pass;
- repaired area <=5%;
- post-repair target-carrier loss <=10%;
- mask tiles periodically without seams or isolated edge fragments.

### M2 optical gate

- minimum carrier visibility >=0.15 at nominal conditions after spectrum, source, and detector MTF;
- carrier balance CV <=0.10;
- local carrier contamination <=0.05;
- a connected operating window covers at least +/-5% distance and +/-3% energy;
- at least 90% of joint tolerance samples meet all optical gates.

### M3 reconstruction gate at 10,000 photons/pixel

- phase NRMSE <=10% of truth RMS;
- transmission RMSE <=5e-4;
- D0 relative RMSE <=5%;
- anisotropy RMSE <=0.05;
- D-angle MAE <=5 degrees;
- effective D spatial resolution FWHM <=10 um;
- no headline method uses truth-dependent offset, filter, or hyperparameter selection.

At 3,000 photons/pixel, D-angle MAE must be <=10 degrees. Results must include 95% confidence
intervals and remain valid under model-mismatch phantoms.

### M4 superiority gate

The selected new design must beat both the corrected S4 and the best established baseline (CUC, HPG,
RBN, or MPG as applicable) by at least 15% in the preregistered composite score, or require at least
20% fewer photons for matched phase/D accuracy and resolution. If it only beats grid/checkerboard, the
novelty claim is too weak.

## 10. Test-coupon fabrication plan (Milestone M5)

Fabricate a small coupon before a full-field device. Put all controls on the same wafer and use the
same etch:

```text
S4 corrected star
H3 binary
H3 four-level or two-depth test cells
CUC/hex-packed circular baseline
grid and checkerboard controls
CD variants: nominal, -0.25 um, +0.25 um
depth variants: nominal, -5%, +5%
```

Recommended first material is silicon at the selected monochromatic design energy. Before exposure,
measure CD, depth, sidewall angle, roughness, missing features, and across-field uniformity using SEM,
profilometry/interferometry, and suitable X-ray metrology. Update the simulator with measured geometry
before comparing prediction and experiment.

**Coupon gate:** measured target-carrier visibility must be at least 80% of the predicted value and all
target carriers must remain separable. Measured depth must be within +/-3% and median CD within
+/-0.2 um of the accepted layout, or the process must be corrected before beamline imaging.

## 11. Experimental validation plan (Milestone M6)

### 11.1 Calibration measurements

1. Dark image and detector gain map.
2. Reference images at several energies and distances around nominal.
3. Repeat references over time to measure drift.
4. Flat-field Fourier spectrum, per-carrier visibility, phase, and contamination.
5. Edge/point target for effective phase/T/D spatial resolution.

### 11.2 Samples

- calibrated weak-phase object with known thickness/material;
- absorption step or wedge;
- isotropic scattering standard;
- aligned carbon-fibre bundles at known angles;
- crossed-fibre and two-orientation samples;
- one practical material sample relevant to the intended application.

### 11.3 Experimental comparison

Acquire star/H3/baseline data with the same incident dose, energy, geometry, detector, and illuminated
area. Randomize acquisition order to reduce drift bias. Use blinded sample labels during reconstruction.
Report phase NRMSE, T RMSE, D0 error, anisotropy error, angle error, effective resolution, acquisition
time, and dose. Publish failed runs and calibration exclusions according to preregistered rules.

## 12. Decision after the coupon experiment (Milestone M7)

Proceed to a full-area grating only if:

- M0-M6 gates all pass;
- measured ranking agrees with simulation within confidence intervals;
- the candidate beats a meaningful published baseline, not only a rank-deficient grid;
- the result supports a narrow novelty claim that survives a formal literature and patent review;
- fabrication yield and field uniformity are sufficient for the required field of view.

If H3 beats S4 but not CUC/RBN/MPG, publish the optimization/negative comparison if scientifically
useful, but do not claim a superior new grating. If no periodic design meets the bandwidth or
fabrication gates, pivot to the random/blue-noise modulator track rather than relaxing the gates.

## 13. Recommended execution order and estimated schedule

| phase | work | expected duration | primary output |
|---|---|---:|---|
| 1 | M0 physical material and independent forward model | 1-2 weeks | validated physical reference images |
| 2 | M1 DRC and candidate library | 1-2 weeks | fabrication-safe S4/H3/CUC/RBN masks |
| 3 | M2-M3 Pareto optimization and frozen validation | 2-4 weeks | ranked Pareto front with confidence intervals |
| 4 | M4 dense tolerance and Monte Carlo | 1-2 weeks | operating windows and failure attribution |
| 5 | design review and coupon layout freeze | 1 week | signed-off GDSII and preregistration |
| 6 | coupon fabrication and metrology | process dependent, typically 4-8 weeks | measured geometry and flat-field prediction |
| 7 | M6 beamline/lab experiment | 1-2 weeks plus access time | calibrated phase/T/D comparison |
| 8 | M7 full-area go/no-go | 1 week | fabrication and publication decision |

## 14. Required reports and figures

Every candidate-selection run must produce:

1. physical mask and GDS audit view;
2. material thickness/absorption report;
3. reference intensity and Fourier spectrum with labelled carriers;
4. per-carrier visibility, balance, contamination, and Fisher eigenvalues;
5. energy-distance operating-window map after MTF and spectrum;
6. fabrication-tolerance tornado plot and joint Monte Carlo pass rate;
7. phase/T/D truth-reconstruction-error maps;
8. D0/anisotropy/angle error versus dose at matched resolution;
9. Pareto plot of accuracy, dose, coverage, and fabrication complexity;
10. final go/no-go table listing every gate and supporting artifact.

## 15. Novelty discipline and literature baselines

Single-shot multidirectional X-ray phase/dark-field imaging, circular unit-cell gratings, random-mask
speckle imaging, modulated phase gratings, and hexagonal directional patterns in related modalities
already exist. The project must not claim those broad ideas as new.

The defensible novelty paths are:

- H3 as an X-ray phase carrier design that achieves rank-3 directional D with fewer carriers and a
  measured visibility/dose benefit;
- DRH6 as a single-shot directional and multi-correlation-length grating;
- fabrication-aware Fisher-information topology optimization with experimentally validated tolerance;
- a demonstrated regime of lower detector-resolution demand, wider bandwidth, or lower dose than
  circular-cell and random-mask baselines.

Before layout freeze, perform a formal paper and patent search using the exact carrier geometry,
symmetry, phase levels, and intended reconstruction claims. Treat novelty as unproven until that review
and the superiority gate both pass.

Primary literature baselines for implementation and comparison:

- 2-D single-shot transmission grating imaging: https://pmc.ncbi.nlm.nih.gov/articles/PMC3091831/
- circular unit-cell and hexagonal packing: https://doi.org/10.1088/1742-6596/849/1/012053
- omnidirectional single-shot hard-X-ray scattering: https://doi.org/10.1103/PhysRevLett.116.093902
- directional single-grid retrieval: https://pubmed.ncbi.nlm.nih.gov/37155790/
- random/speckle directional D: https://arxiv.org/abs/2105.10693
- modulated phase grating experiments: https://arxiv.org/abs/2404.14584
- spectral/correlation-length D: https://pmc.ncbi.nlm.nih.gov/articles/PMC7411057/
- 2-D grating fabrication: https://doi.org/10.1016/j.mee.2012.08.025
