# model-benchmark

# Skill 审查报告

**Skill 名称**：model-benchmark (miemie-llm-benchmark) **审查日期**：2026-04-15 **审查依据**：Skill 审查规范 v1.0 **Skill 类型**：工具调用型（CLI/Python 脚本执行）

---

## 审查总览

| 维度 | 检查结果 | 备注 |
| --- | --- | --- |
| 一、文件结构与命名规范 | 部分通过 | 文件夹名与 YAML name 不匹配 |
| 二、YAML Frontmatter 格式校验 | 部分通过 | name 与文件夹名不一致 |
| 三、Description 触发质量 | 通过 | 触发词丰富，但缺少负面触发和局限性声明 |
| 四、SKILL.md 内容完整性 | 未通过 | 缺少多个必需章节 |
| 五、集成度 | 通过 | 使用环境变量存储 API Key，无需单独登录 |
| 六、健壮性 | 未通过 | 文件总大小严重超标（4439 行） |
| 七、安全性 | 通过 | API Key 本地存储，日志脱敏处理 |
| 八、合规性 | 部分通过 | 缺少免责声明 |
| 九、可观测性 | 部分通过 | 缺少使用统计机制 |

---

## 严重问题（必须修改）

### 1. 【严重】文件夹名与 YAML name 不匹配

**问题描述**：

*   SKILL.md 中的 YAML name 为 `model-benchmark`
    
*   实际文件夹名为 `miemie-llm-benchmark`
    

**规范依据**：规范 2.1 明确规定"Skill 名称必须与文件夹名保持一致"

**修改建议**： 将文件夹名从 `miemie-llm-benchmark` 修改为 `model-benchmark`，或在 SKILL.md 中将 name 修改为 `miemie-llm-benchmark`。

---

### 2. 【严重】SKILL.md 缺少必需章节

**问题描述**： 当前 SKILL.md 缺少以下规范要求的必需章节：

| 缺失章节 | 规范要求 | 当前状态 |
| --- | --- | --- |
| 安全红线 | 规范 4.1 要求声明安全边界 | 仅在正文中提及 API Key 处理，无独立章节 |
| MCP 工具清单 | 规范 4.1 要求声明使用的工具 | 未声明使用 CLI/Shell 工具而非 MCP |
| 场景识别与工具选择 | 规范 4.1 要求声明场景匹配逻辑 | 有工作流描述但无正式章节 |
| 终止与总结 | 规范 4.1 要求声明结束条件 | 有报告要求但无正式章节 |

**修改建议**： 在 SKILL.md 中补充以上四个必需章节，明确声明：

1.  安全红线：API Key 本地存储、日志脱敏、禁止上传敏感数据等
    
2.  MCP 工具清单：明确声明"本 Skill 不使用 MCP 工具，通过 CLI/Python 脚本执行"
    
3.  场景识别与工具选择：声明触发条件和场景匹配逻辑
    
4.  终止与总结：声明任务完成条件和输出交付标准
    

---

### 3. 【严重】文件总大小严重超标

**问题描述**：

*   **当前总大小**：4439 行
    
*   **规范限制**：所有附属文件总大小不应超过 500 行（规范 6.5）
    
*   **超标倍数**：约 8.9 倍
    

**文件分布详情**：

| 文件路径 | 行数 | 备注 |
| --- | --- | --- |
| scripts/model\_benchmark.py | 2269 | 严重超标（建议不超过 100 行） |
| references/evalscope-official/SKILL.md | 425 | 超标 |
| references/evalscope-official/perf-reference.md | 328 | 超标 |
| references/evalscope-official/eval-reference.md | 232 | 超标 |
| references/evalscope-official/examples.md | 226 | 超标 |
| tests/test\_model\_benchmark.py | 332 | 超标 |
| scripts/patch\_evalscope.py | 329 | 超标 |
| configs/model\_benchmark.example.yaml | 87 | 正常 |
| SKILL.md | 114 | 正常 |
| 其他脚本文件 | <50 | 正常 |

**规范依据**：规范 6.5 "所有附属文件总大小超过 500 行时，必须评估上下文窗口占用。单个文件超过 100 行时，建议拆分为多个文件或使用外部引用"

**修改建议**：

1.  **精简参考文档**：将 references/ 目录下的官方文档改为链接或摘要形式，不要完整复制
    
2.  **拆分主脚本**：将 model\_benchmark.py（2269 行）按功能拆分为多个小模块
    
3.  **移除测试文件**：tests/ 目录不应包含在 Skill 中，建议移至独立仓库
    
4.  **评估必要性**：重新审视每个文件是否必须包含在 Skill 中
    

---

## 建议优化（非阻塞）

### 1. 【建议】补充负面触发声明

**问题描述**：Description 中缺少 negative triggers（负面触发）声明，可能导致误触发。

**修改建议**： 在 description 中增加 `<!-- negative triggers: ... -->` 注释，声明不应触发本 Skill 的场景，例如：

*   非 API 模型的本地模型压测
    
*   仅查询模型信息而不进行压测
    
*   通用性能优化咨询（非压测场景）
    

---

### 2. 【建议】补充局限性声明

**问题描述**：Description 中未声明 Skill 的能力边界和局限性。

**修改建议**： 在 description 中增加 `<!-- insufficiency: ... -->` 注释，声明本 Skill 的局限性，例如：

*   仅支持 HTTP API 形式的模型服务，不支持本地模型
    
*   压测结果受网络环境影响
    
*   不支持自定义评估指标
    

---

### 3. 【建议】补充平台声明

**问题描述**：SKILL.md 中仅在 Troubleshooting 章节提及 macOS 和 Ubuntu，缺少顶层的平台兼容性声明。

**修改建议**： 在 SKILL.md 开头增加平台声明章节，明确支持的操作系统和环境要求：

*   支持 macOS（需安装 GNU coreutils）
    
*   支持 Ubuntu/Debian Linux
    
*   需要 Python 3.8+
    
*   需要安装 evalscope 依赖
    

---

### 4. 【建议】补充输出免责声明

**问题描述**：生成的性能报告缺少免责声明，可能被误解为官方认证结果。

**修改建议**： 在报告生成章节增加免责声明，例如： "本报告仅供参考，实际生产环境性能可能因网络、硬件、负载等因素存在差异。"

---

### 5. 【建议】增加使用统计机制

**问题描述**：缺少使用统计机制，无法追踪 Skill 的使用情况。

**修改建议**： 考虑增加轻量级的使用统计，例如：

*   记录压测执行次数
    
*   记录常用模型类型
    
*   记录平均执行时长
    

---

## 检查规范的改进建议

本次审查未发现规范本身的明显缺陷。针对本 Skill 的特殊情况，建议：

1.  **对于大型工具型 Skill**：规范 6.5 关于文件大小的限制可能需要更灵活的解释。对于必须包含大量代码的工具型 Skill，建议：
    
    *   允许将核心脚本作为"黑盒"引用，不在上下文中展开
        
    *   要求提供脚本的功能摘要而非完整代码
        
    *   将详细文档移至外部链接
        
2.  **对于参考文档**：建议明确区分"必需包含的文档"和"可选引用的文档"，避免将完整的第三方文档复制到 Skill 中。
    

---

## 附录：文件清单

| 序号 | 文件路径 | 行数 | 状态 |
| --- | --- | --- | --- |
| 1 | SKILL.md | 114 | 正常 |
| 2 | scripts/model\_benchmark.py | 2269 | 严重超标 |
| 3 | scripts/patch\_evalscope.py | 329 | 超标 |
| 4 | scripts/convert\_dataset.py | 32 | 正常 |
| 5 | scripts/generate\_report.py | 32 | 正常 |
| 6 | scripts/sla\_autotune.py | 19 | 正常 |
| 7 | scripts/benchmark.sh | 7 | 正常 |
| 8 | scripts/stability\_test.sh | 7 | 正常 |
| 9 | references/evalscope-official/SKILL.md | 425 | 超标 |
| 10 | references/evalscope-official/eval-reference.md | 232 | 超标 |
| 11 | references/evalscope-official/examples.md | 226 | 超标 |
| 12 | references/evalscope-official/perf-reference.md | 328 | 超标 |
| 13 | configs/model\_benchmark.example.yaml | 87 | 正常 |
| 14 | tests/test\_model\_benchmark.py | 332 | 超标 |
| 15 | .gitignore | \- | 正常 |
| **总计** |  | **4439** | **严重超标** |

---

_报告生成时间：2026-04-15_

_审查工具：Skill 审查规范 v1.0_