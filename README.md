# basalt-arm-sm121

**Basalt 推理引擎的 Arm64 / sm_121 移植版。** 让 Qwen3.8-Flash-Next 和 Qwen3.8-27B 在 NVIDIA RTX Spark N1X 这类
Arm64 + 集成 Blackwell GPU（计算能力 12.1，和 DGX Spark 的 GB10 同代）的机器上原生运行，Windows on Arm64 和 Linux Arm64 都能编译。

*English:* a port of the [Basalt](https://github.com/jesdga95/basalt) inference engine to Arm64 CPUs and compute
capability 12.1 GPUs (RTX Spark N1X, DGX Spark / GB10), building natively on Windows on Arm64 (MSVC) and on Linux Arm64.
On an RTX Spark N1X it decodes Qwen3.8-Flash-Next about 2.1 times as fast as vLLM on the same requests. **This fork
also adds the first verified DGX Spark (GB10) run of Flash-Next MoE, plus a Linux unified-memory fix**; see the DGX Spark
section below. Build guide, design notes, test status and measurements: [docs/arm64.md](docs/arm64.md). The upstream README is
kept as [README.basalt.md](README.basalt.md).

## 这是什么

- **上游**：[jesdga95/basalt](https://github.com/jesdga95/basalt)（MIT 许可），一个面向 NVIDIA Blackwell 显卡的本地大模型推理引擎，
  支持 MoE 专家分层存放、MTP / 块草稿推测解码、OpenAI 和 Anthropic 兼容接口。上游只支持 x86-64 CPU + sm_120 独立显卡。
- **本仓库**：在上游最新代码（`63cb68a`，v0.4.1 之后）的基础上，加上 Arm64 CPU 和计算能力 12.1 集成显卡的支持，独立维护，
  不向上游提交。上游文件只做最少的机械性改动，新逻辑尽量放在新文件里，方便以后跟进上游更新。
- **实测机器**：一台 RTX Spark N1X 笔记本（128 GB 统一内存，其中 96 GB 划给 GPU），Windows 11 on Arm 原生运行，
  以及同一台机器上的 WSL2（Ubuntu 26.04 aarch64）。**DGX Spark（GB10）也已在本 fork 上实测通过**——见下文
  「DGX Spark（GB10）实测结果」一节。原移植版 README 曾把 DGX Spark 列为「未实测」，本 fork 补上了这一格。

## 相比上游做了什么

**Arm64 编译**
- Linux Arm64（GCC）和 Windows on Arm64（MSVC 原生）都能完整编译；一次编译对应一种 CPU 架构和一种计算能力，x86-64 + 12.0 保持不变。
- CPU 内核换成 NEON 版本：各种量化格式的点积由 llama.cpp 的 Arm 内核生成（MIT），NVFP4、多 token 的 Q2_0、路由和视觉 GEMM 是自己写的。
- 内存顺序：Arm64 的自旋等待、读写屏障和预取都有对应实现，轮询 GPU 写的环形队列后加读屏障。

**集成显卡（N1X）的特殊处理**
- **显存读数**：`cudaMemGetInfo` 会把共享内存段也算进总量，而且看不到通过虚拟内存接口映射的显存。改成以划给 GPU 的那块内存为总量，
  按系统级统计算已用量，所有容量规划都用这个读数。
- **Windows 提交量护栏**：集成显卡的每次显存分配都会立刻计入 Windows 的提交量（物理内存 + 页面文件）。引擎会给系统留足余量，
  避免把别的程序挤崩；并建议把页面文件设到足够大（见下文）。
- **上传数据迟到**：从可分页内存同步拷贝到显卡后，数据可能还没真正到达；在集成显卡上拷贝后多做一次同步。
- 跳过 NVML 里列出的 NPU，锁页内存预算改用 DXGI 的本地段，Modern Standby 机器上运行时保持唤醒。

**工具**
- `tools/port/fetch_cuda.py`：从 NVIDIA 的可再分发包拼装 CUDA 工具包（不用安装程序，逐个校验 SHA-256）。
- `tools/port/env-win-arm64.cmd`、`build-win-arm64.cmd`：Windows on Arm64 一键配置和编译。
- `tools/serve_speed.py`：服务端测速（长上下文；把 basalt-bench 的同一批题发给 vLLM 等其他引擎做对比）。

## 实测速度（RTX Spark N1X，Windows 11 on Arm）

Qwen3.8-Flash-Next（`NVFP4-DQ6_K` 模型包），basalt-bench 的同一批题，vLLM 为社区版 vLLM 0.32.0（NVIDIA 的 NVFP4 权重，MTP 草稿 2）：

| | Basalt | vLLM |
| --- | ---: | ---: |
| 解码：技术讲解（conv） | 80.8 tok/s | 49.1 tok/s |
| 解码：JSON（struct） | 140.3 tok/s | 57.9 tok/s |
| 解码：散文（prose） | 67.8 tok/s | 45.6 tok/s |
| 解码：按 token 加权 | **约 116 tok/s** | 约 55 tok/s |
| 预填充 16k / 64k | 2,558 / 2,638 tok/s | 1,753 / 1,780 tok/s |

长上下文（服务端，int8 KV，一段中文长文本，关闭思考）：

| 提示词长度 | 预填充 | 解码 |
| --- | --- | --- |
| 4,128 | 1,473 tok/s | 63.6 tok/s |
| 191,231 | 2,387 tok/s | 55.1 tok/s |
| 260,124 | 2,224 tok/s（117 秒） | 54.3 tok/s |

4 个并发槽位：1 / 2 / 4 路同时解码，总吞吐 63.7 / 78.4 / 80.1 tok/s。

Qwen3.8-27B（`QUASAR-NVFP4` 模型包，稠密模型）：解码 conv 50、struct 157、prose 36 tok/s（关掉草稿 16.5 tok/s，已接近内存带宽上限）；
预填充 16k 2,757 tok/s；260k 长提示词预填充 1,036 tok/s、解码 14.3 tok/s；4 路并发总吞吐 72.3 tok/s。

完整数据和测试方法见 [docs/arm64.md](docs/arm64.md#speed-on-the-rtx-spark-n1x)。

## DGX Spark（GB10）实测结果（本 fork 新增）

移植作者只实测了 RTX Spark N1X，原 README 一直把 DGX Spark 列为「未实测」。本 fork 在一台 DGX Spark 上完成了首次
Flash-Next MoE 实测，并修复了一处 Linux 统一内存下的显存判定问题。补丁已作为
[PR #1](https://github.com/yanmindi/basalt-arm-sm121/pull/1) 提交给移植作者，改动只有 `src/core/device_memory.cpp`，约 40 行。

**环境**：DGX Spark（GB10，ARM64，sm_121，128 GiB 统一内存），Ubuntu 24.04，CUDA 13.0。用
`cmake -DBASALT_MODELS=qwen4exp` 编译；ctest 118/119（唯一失败项是已知的 `video_test`）。

**开箱失败与根因**：用常规参数启动会报 `no VRAM is left for the expert cache: needs 285 slots (751 MiB),
about 32727 MiB more than this card has free`。原因是 DGX Spark 上 NVML 读不到独立显存（`nvidia-smi` 显示 N/A、
`dedicated == 0`），引擎退回用 `cudaMemGetInfo` 的裸读数；Linux UMA 下这个读数把可回收的页缓存算进 used，
只报 13.73 GiB 空闲，而系统 `MemAvailable` 实为 118 GiB。引擎据此把 64.3 GiB 的 RAM 专家外推成约 45 GiB 的
虚账显存占用，专家缓存被算成 0 slots，连锁导致 verify 失败。补丁在「Linux + 集成显卡 + 无 NVML carve-out」这一档
改用 `MemAvailable` 规划缓存；其他平台不受影响，`BASALT_UMA_CF=0` 可还原旧行为，`BASALT_UMA_CAP_GIB=N` 可限幅。

**补丁后加载**（启动参数与失败时完全一致）：

```sh
loaded   68.4GiB 7.5s
experts live in VRAM 63.3 GiB (100%) ... file (SSD) 0.0 GiB (0%)
```

24,576 个专家 100% 驻留显存、零 SSD 读取，verify 的 1–8 token 窗口全部通过。这是移植作者此前只在 N1X 上见过的形态。

**实用的 262K 配置**：

```sh
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
bin/basalt serve models/Qwen3.8-FN-NVFP4-DQ6_K.basalt \
  --max-context 262144 --parallel 4 --slot-full-context --kv int8 \
  --port 18190 --alias Qwen3.8-Flash-Next \
  --python /path/to/venv/bin/python
```

`--parallel` 默认是 1（多出来的请求只会排队，不是真并发），要让多个流各有完整上下文必须加 `--slot-full-context`。
注意边界：4 个槽位若真的同时用满 262K（共 1,048,576 cells），KV 增长会把专家缓存挤到约 292 slots，专家开始换出；
短上下文 agent 不受影响。

单流 decode（服务端 `predicted_per_second`，temp=0，256 tokens，每场景 2 次）：

| 场景 | 8K 上下文 | 262K 上下文 |
| --- | ---: | ---: |
| chat | 59.8 tok/s | 58.5 tok/s |
| code | 83.6 tok/s | 82.0 tok/s |
| agent（tool_calls） | 79.2 tok/s | 78.5 tok/s |

从 8K 提到 262K 几乎不掉速。

**与现有 vLLM 方案的 A/B**（同 prompt、temp=0、256 tokens、每组合 2 次；两引擎内存不能共存，串行测试；
表内是端到端总吞吐 = 生成 token / 墙钟，含 prefill——这是两边唯一口径一致的指标）：

| 场景 | 并发 | 本方案（Basalt 262K） | vLLM | 倍率 |
| --- | ---: | ---: | ---: | ---: |
| chat | 1 | **52.4** | 19.7 | 2.66x |
| chat | 2 | **69.1** | 31.5 | 2.19x |
| chat | 4 | **74.2** | 44.7 | 1.66x |
| code | 1 | **71.5** | 33.1 | 2.16x |
| code | 2 | **81.6** | 44.3 | 1.84x |
| code | 4 | 74.7 | **114.5** | **0.65x** |
| agent | 1 | **51.7** | 25.5 | 2.03x |
| agent | 2 | **63.5** | 43.6 | 1.46x |
| agent | 4 | 54.3 | **87.0** | **0.62x** |

结论是分化的：单流和低并发本方案约 2 倍领先；但 4 并发短输出时 vLLM 反超——单流已接近 LPDDR5X 带宽天花板，
并发主要是在分摊带宽，而 vLLM 的连续批处理把 4 个短请求合成一轮。**不要**把表里的单流 decode 数字拿去和别人的
纯 decode 数字比（这里含 prefill），也不要读成「Basalt 普遍快 2 倍」。

已知瑕疵：agent 的原生 tool call 在 `--parallel 4` 下出现过间歇性 HTTP 400（`arguments` 被输出成 JSON 对象而非字符串），
补测 14 个请求全部 200，属间歇现象；并发时各流速度不均（服务端稳态日志显示每流约 17–21 tok/s、合计约 77），看总吞吐即可。

## 快速开始：Windows on Arm64

需要：Visual Studio 2026 或其 Build Tools（勾选 ARM64 C++ 工具），Python 3.10+（Arm64 版），足够大的页面文件（见下文）。

```bat
git clone https://github.com/yanmindi/basalt-arm-sm121 && cd basalt-arm-sm121
python tools\port\fetch_cuda.py windows-arm64 C:\cuda-13.5
tools\port\build-win-arm64.cmd build C:\cuda-13.5
```

下载模型包（任选其一；国内可以设置 `HF_ENDPOINT=https://hf-mirror.com` 走镜像）：

```bat
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-NVFP4-DQ6_K.basalt --local-dir models
hf download jesdga/Qwen3.8-27B-Basalt Qwen3.8-27B-QUASAR-NVFP4.basalt --local-dir models
```

启动服务（OpenAI / Anthropic 兼容接口，默认 `http://127.0.0.1:8080`，浏览器打开同一地址是统计页面）：

```bat
set PATH=C:\cuda-13.5\bin\arm64;%PATH%
pip install -r requirements.txt
bin\basalt.exe serve models\Qwen3.8-FN-NVFP4-DQ6_K.basalt --python python.exe --max-context 262144 --kv int8
```

`--python` 填装好依赖的那个 Python 解释器的路径。`bin\arm64` 一定要在 PATH 里，否则 Windows 会弹出"找不到 cublas64_13.dll"的窗口。

## 快速开始：Linux Arm64

```sh
python3 tools/port/fetch_cuda.py linux-sbsa ~/cuda-13.5
export PATH=~/cuda-13.5/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
pip install -r requirements.txt
LD_LIBRARY_PATH=~/cuda-13.5/lib bin/basalt serve models/Qwen3.8-FN-NVFP4-DQ6_K.basalt --max-context 262144 --kv int8
```

WSL2 下还要把 `/usr/lib/wsl/lib` 加进 `LD_LIBRARY_PATH`。

## 使用注意

- **页面文件**：在 Windows 上，集成显卡的显存会计入提交量。128 GB 的 N1X 建议把页面文件设为自定义大小，**至少约 72 GB**
  （我们用的是 81,920～98,304 MB）。系统自动管理的 68 GB 页面文件下，引擎为了安全会少放一些专家进显存，每个请求慢约 0.9 秒。
- 一次只跑一个大模型：Flash-Next 加载后占 70 GB 以上显存，不要和 vLLM 等其他推理服务同时运行。
- **测试状态**：Linux Arm64 225 项过 224 项，Windows on Arm64 219 项过 213 项。没过的每一项及原因见 [docs/arm64.md](docs/arm64.md#tests)。
- **没有实测过**：DeepSeek-V4.1-Flash 的真实权重（模型包 260 GB 以上）、真正进入待机、图片和视频输入、
  在全新的 Arm64 Python 环境里安装依赖、本移植之后的 x86-64 编译。（DGX Spark 已在本 fork 实测，见上文。）

## 文档

- [docs/arm64.md](docs/arm64.md)：Arm64 编译、集成显卡的处理、测试状态、实测数据（英文）。
- [README.basalt.md](README.basalt.md)：上游原版说明，支持的模型、配置文件、API、视频等用法仍然适用。
- [docs/](docs/)：上游的完整文档（运行、配置、模型、精度、模型文件格式）。

## 许可与致谢

- 本仓库沿用上游的 MIT 许可（[LICENSE](LICENSE)），第三方代码（ggml / llama.cpp）的许可见 [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES)。
- Basalt 由 [jesdga95](https://github.com/jesdga95) 开发，模型包来自其 Hugging Face 页面。
- Windows on Arm64 部分参考了 SolettaSolaris 在上游 PR [jesdga95/basalt#230](https://github.com/jesdga95/basalt/pull/230) 中的思路，没有复用其代码。
- 本仓库与上游项目和 NVIDIA 均无隶属关系。
