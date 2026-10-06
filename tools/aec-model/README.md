# AEC 模型本地制品工具

维护者使用的独立工具：**作者原始 TFLite → 固定转换 → VoiceInsight ONNX 制品 → 显式 GitHub Draft Release**。

本工具本身不修改已安装模型、AEC算法或运行时。工具与应用源码仓库（VoiceInsight）已**完全分离**：本仓库只放转换/复验/发布工具与版本化制品，不含应用源码、Python环境或私有数据；应用侧只保留原生 Rust 下载器，接入本仓库的公开 Release，显式支持128/256/512。没有 GitHub Actions/远程构建，工具不会自动公开 Release，也不自动提交或推送任何仓库。

## 信任与范围

- 唯一模型上游：`breizhn/DTLN-aec`，固定提交 `9d24e128b4f409db18227b8babb343016625921f`。两份作者权重及MIT许可按固定大小/SHA-256验证。
- 生成制品不需要 anarlog、应用安装目录、实验目录中的模型或会议数据。原始模型不放进Git。
- `spec.json` 固定接口、opset、工具版本、作者来源及经过先前独立评估的权重/计算图指纹。这个静态基准是显式版本契约，不自动从上游最新文件重新学习。这四个工具文件（`spec.json`、`builder.py`、`pyproject.toml`、`uv.lock`）的 SHA-256 也被写进每个已发布制品的 `manifest.json`，必须逐字节保持，不能为了迁移或“整理”而重写。纯标准库的 `release.py` 与 `cli.py` 刻意不在这四个文件里，可以直接演进。
- `pyproject.toml` 固定直接依赖；`uv.lock` 锁定完整依赖解析及下载文件哈希。构建使用 `uv sync --locked --no-dev --no-build`，不运行sdist构建、不自动更新锁文件。
- 第一版维护者环境限定 **macOS arm64、uv 0.12.22、Python 3.11.15**。工具不自动升级全局uv；uv可准备指定Python及工具目录下的私有 `.venv`。这不是应用/ONNX制品仅支持macOS的声明；其他构建平台尚未验证。
- ONNX是**VoiceInsight维护的转换制品**，不是作者直接发布的“官方ONNX”。保留 `DTLN-LICENSE`，发布包带原始许可和明确的NOTICE。

## 1. 生成与验证

在本仓库根目录执行。`--variant` 必填，只接受 `128`、`256`、`512`，构建、复验、发布都必须显式传入；文件名、状态形状和发布tag按规格隔离，不跨规格推断或回退：

```bash
python3 tools/aec-model/cli.py build --variant 256 --version 1.0.0
```

流程：创建独立本地运行目录 → 锁定Python环境 → 下载三份固定公开文件 → SHA/大小校验 → 转换两阶段ONNX → 模型门禁 → 生成manifest → 再次读取实际制品验证 → 完整bundle就绪。

模型门禁包含：

1. ONNX checker、opset、输入输出名称/shape/dtype；拒绝外置tensor数据。
2. 全部参数、算子、属性、拓扑的规范化指纹，与已经评估的作者模型一致。
3. 每个阶段120帧、两次显式状态重置的真实ORT推理；输出必须有限且非静音，reset误差≤1e-6。
4. 固定的作者输入来源、工具文件哈希、制品大小/SHA。

输出示例（实际路径由命令打印）：

```text
test-results/aec-model-<time>-<uuid>/
├── summary.json
├── build.log
├── environment-setup.log
├── conversion-1.log / conversion-2.log
├── sources/                     # 原始作者文件，仅保存在本地
└── bundle/                      # 唯一允许发布的目录
    ├── model_256_1.onnx
    ├── model_256_2.onnx
    ├── manifest.json
    ├── LICENSE-DTLN-aec
    ├── NOTICE
    └── validation-summary.json
```

每次构建新建目录，不覆盖旧制品或应用模型。失败保留日志/部分staging作为证据；没有验证完成的bundle不能发布。ONNX内部自动命名可能使两次构建的文件SHA不同；同一发布版本必须冻结具体制品，不能重建后覆盖同版本资产。

### 跨运行时失败不会被隐藏

`validation-summary.json` **区分必需ONNX制品门禁和跨运行时诊断**。本工具不切换到TFLite引擎：

- ONNX图/权重/接口/状态门禁失败：构建失败，不产生可发布bundle。
- TFLite/XNNPACK vs ONNX的严格 `1e-4` 输出/内部状态诊断：如实保存PASS/FAIL，不提高阈值。已知第一阶段state误差可达 `1.220703125e-4`，因此报告仍含FAIL。
- 这不是完整声学质量验收，更不表示现有回声消除失败已修复；没有在构建中重新跑12场景ASR/真实设备/完整all。

`build: PASS`仅表示明确列出的ONNX制品门禁通过。发布要求显式确认这些局限，不允许把报告笼统宣传为“全部测试通过”。

## 2. 独立复验

```bash
python3 tools/aec-model/cli.py verify --variant 256 --version 1.0.0 --directory /absolute/path/to/bundle
```

要求已有锁定的工具环境；**不下载文件、不重新转换、不导入TensorFlow**。它会读取真实ONNX，重新核验图/参数、接口和循环状态，而非只相信JSON中的 `mandatoryGatesPassed`。修改参数后即使同步伪造文件哈希和报告哈希，也会被固定语义指纹拒绝。

版本与manifest必须匹配；构建工具文件发生变化时也拒绝旧receipt。应使用原工具版本验证/发布，或在新版本下重新构建评估，而不是手改manifest。

## 3. 发布预检（零GitHub请求）

```bash
python3 tools/aec-model/cli.py publish \
  --variant 256 \
  --version 1.0.0 \
  --directory /absolute/path/to/bundle \
  --repo quyao/voiceinsight-models \
  --target <完整40位Git提交SHA> \
  --dry-run
```

`--target`须由维护者明确选择，建议使用包含已审核制品工具的提交；不默认取远端分支的最新提交。本工具不会帮你提交或推送当前未提交改动。

Dry-run不需要GitHub认证，也不调用gh或GitHub。它在私有临时快照上做实际ONNX复验，记录计划发布的repo、tag、六个资产的SHA/大小、诊断及局限。结果在本轮 `summary.json` 中，`networkCalls:0`。**它不能证明远端权限、标签空闲或上传服务可用。**

## 4. 显式上传为 Draft

确认预检、模型许可/NOTICE、目标仓库与提交后，维护者使用自己已配置的GitHub CLI认证：

```bash
python3 tools/aec-model/cli.py publish \
  --variant 256 \
  --version 1.0.0 \
  --directory /absolute/path/to/bundle \
  --repo quyao/voiceinsight-models \
  --target <完整40位Git提交SHA> \
  --confirm-draft \
  --acknowledge-limitations
```

两个确认参数缺一不可。发布阶段：

1. 固定六文件白名单，拒绝额外文件、symlink、超大文件、损坏文件、伪造/不匹配报告。
2. 复制本地快照，实际ONNX复验并在验证后再次确认文件未变。
3. 检查仓库写权限、目标提交存在、release/tag都不存在。
4. 创建新的 `aec-dtln-<variant>-v<version>` **Draft**，`make_latest:false`。
5. 按本次创建的 **release ID** 上传，不按可能被重新绑定的tag定位资产，不使用clobber。
6. 上传前后确认仍为相同Draft，校验GitHub返回的每个资产size和SHA-256 digest。
7. 不自动公开。维护者需要在GitHub中另行审核并公开。

已有Release或Tag直接拒绝，不覆盖、不自动续传。鉴权/网络/服务错误不能当作“版本不存在”。上传或远端校验失败后保留Draft及本地记录，不自动删除Release/Tag，不重试到通过；人工检查后决定删除该Draft或使用新版本。创建请求结果不明确时也应先检查远端，不能盲目重试。

只支持github.com：凭据由gh从本机认证或对应GitHub环境变量读取；只交给GitHub子进程，发送至GitHub API/上传端点。不把凭据放进参数、文件、日志或发布资产；不转发其他Provider凭据或GH_DEBUG。源模型构建阶段不接收GitHub凭据。

## 5. 运行与测试纪律

- 构建/验证/发布在本仓库 `test-results/` 下持有自己的运行锁，遇到已有运行立即BLOCKED，不删除别人持有的锁，也不干扰任何其他仓库的测试运行。
- 下载大小和超时有界；转换、环境安装和API/上传子进程均有超时。中断/超时会清理自有进程组和锁，不终止其他任务；本地失败产物保留。
- 发布快照仅含六份公共制品，不上传源码缓存、日志、安装环境、音频、转录或应用私有数据。
- 本工具生成/发布模型不影响正在使用的应用模型。应用通过受信任的固定资产清单原生下载，三规格使用各自文件和SHA；安装时持有与模型加载共享的文件锁，普通提交错误恢复旧文件，中断造成的缺失配对报错、不换规格。旧128隐式回退已经移除。

本仓库自带回归，只依赖标准库 `unittest`，不需要模型、网络或项目测试 runner：

```bash
python3 -m unittest discover -s tests -v
```

新增测试覆盖路径/版本/资产边界、checksum与语义复验调用、伪造报告、快照变更、dry-run零调用、已有release/tag、鉴权与权限错误、Draft-only上传、远端digest失败、中断/超时和进程后代清理。发布网络流程使用模拟客户端，不能称为真实GitHub上传验证。

本次实际执行记录见 [VALIDATION.md](VALIDATION.md)。普通应用用户不需要 Python、uv 或 gh 来安装已发布的模型；这三样仅是维护者工具：`python3` 跑标准库入口与回归，`uv` 准备锁定的转换环境，`gh` 只在真实发布时使用。
