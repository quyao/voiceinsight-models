# 本地验证记录 — 2026-10-05

> **迁移说明（2026-10-06）**：本文件与工具一起从应用仓库 `quyao/voiceinsight` 的 `tools/aec-model/` 原样迁入 `quyao/voiceinsight-models`。下文记录的是拆分前在应用仓库中执行的验证：其中的 `npm test -- unit runner`、`npm test -- check` 指的是当时应用仓库的统一测试入口，当时工具回归挂在那个 runner 里。迁移后本仓库用自己的回归覆盖同样的用例；工具文件（`spec.json`、`builder.py`、`pyproject.toml`、`uv.lock`）逐字节未改，SHA-256 与已发布制品 manifest 中的 `toolHashes` 一致。以下历史结论不改写。
>
> **纯 Python 化（2026-10-06）**：原先的 Node 入口与回归（`package.json`、`scripts/aec-model.mjs`、`tools/aec-model/release.mjs`、`tests/aec-model.test.mjs`）已改写为纯标准库 Python（`tools/aec-model/release.py`、`tools/aec-model/cli.py`、`tests/test_aec_model.py`），仓库不再需要 Node/npm；四个被哈希的工具文件保持不变，已发布制品的 `toolHashes` 仍匹配。本轮验证：`python3 -m unittest discover -s tests` 19 项 PASS；对已发布 `aec-dtln-256-v1.0.0` 制品跑真实 `cli.py verify` PASS；`cli.py publish --dry-run` PASS（`networkCalls:0`）。真实 Draft 上传未在本轮执行。

> 下文为最初256单规格工具的历史验证。后续已扩展128/256/512、按用户授权公开发布模型预发布版本并接入应用。历史的“未发布/未接入”描述仅对应本轮早期状态，不代表当前状态。2026-10-05 的 AEC 研究环境与产物已删除，本文只保留检查项和结论。

范围：本地制品工具落地与验证。**没有创建/上传/公开GitHub Release，没有修改应用下载地址或运行时，没有替换已安装模型。**

## 实际执行

| 检查 | 结果 | 说明 |
|---|---|---|
| 作者权重真实下载、锁定环境、两阶段转换 | PASS | 本轮 summary.json |
| 实际制品图/权重指纹、接口、循环状态/reset | PASS | 上述 bundle/validation-summary.json |
| 独立 `model:aec:verify` | PASS | 本轮 summary.json |
| 最新发布dry-run，真实ONNX复验 | PASS，GitHub请求0 | 本轮 summary.json |
| 修改真实ONNX参数并同步伪造文件/报告哈希 | 负对照PASS：被语义门禁拒绝，预期exit1，清理PASS | 构建目录的 negative-weight-tamper 结果/日志 |
| `npm test -- unit runner` | 101项PASS、0失败、0跳过 | 统一 runner 本轮 summary.md |
| `npm test -- check` | 6组PASS：types/config/build/format/rust/clippy | 统一 runner 本轮 summary.md |
| 独立 `npm run check`、`npm run build` | PASS | 本轮终端执行 |
| JS语法、Python编译、`git diff --check` | PASS | 本轮终端执行 |

单元套件含发布模拟：existing release/tag、401/权限拒绝、明确Draft确认、错误的远端digest、发布中Draft变更、失败保留Draft、快照篡改、进程超时/后代清理。不是实际GitHub上传。初轮发现HTTP零header响应解析缺陷，已修复后回归通过；原失败日志已随实验产物清理。

## 没有被改写成通过的诊断失败

本次第一阶段 TFLite/XNNPACK vs ONNX：

- output max absolute error：`0.000002682209014892578`
- recurrent state max absolute error：`0.0001220703125`
- 预定阈值：`0.0001`
- 诊断判定：**FAIL**。

第二阶段诊断PASS。两个阶段的必需ONNX图/参数/接口/状态门禁均PASS。由于工具不切换运行时，跨运行时结果是显式保留的诊断，不是被放宽的阈值或被宣称通过的严格一致性；真实发布要求 `--acknowledge-limitations`。

本次输出版本号为 `1.0.0`，**仅为本地待发布制品，不表示远端已有该版本**。默认应用下载仍然未迁移。

## 未执行 / 未覆盖

- 真实GitHub鉴权、上传和远端资产hash校验：实现和模拟覆盖，未执行live upload；没有公开发布。
- 项目完整 `npm test/all`、新一轮ASR/声学/真实设备/原生UI验收未执行。上述检查均为局部，`fullAcceptancePassed:false`。
- 本轮没有修复已有声学质量失败，也没有切换TFLite引擎。
- 构建和验证环境为macOS arm64；其他维护者平台未验证，工具明确拒绝而非静默兼容。
- 发布目标commit应包含维护者已审核的工具版本；当前工作区有既有未提交改动，本轮没有替用户提交、推送或发布。
