# paper2md — 学术论文转 Markdown MCP 工具

一个 Model Context Protocol (MCP) 工具，将学术论文转换为结构清晰的 Markdown 格式，
包含 YAML 前言和本地图片。支持多种论文来源，针对每种来源采用专属提取策略以实现最高
还原度。

## 工作原理

paper2md 采用针对每种来源优化的**分层提取策略**：

| 来源                   | 主要方法                                      | 说明                                                                                                                     |
| ---------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **arXiv**              | ar5iv HTML → 结构化 DOM 遍历器                | ar5iv = LaTeXML 输出（与 arXiv 官方 HTML 使用相同引擎），已是目前最好的 LaTeX→HTML 转换器                                 |
| **Nature**             | Nature.com HTML → 语义化章节解析器            | Nature 提供标准的 `<section data-title>` 标记，直接解析，零 HTML 残留                                                     |
| **PDF（远程）**        | [MinerU](https://github.com/opendatalab/MinerU) 云端 API（precision v4 或 agent v1） | SOTA 级的 PDF 文档解析。Precision API 需 token，Agent API 无需                                                      |
| **PDF（本地）**        | [MinerU](https://github.com/opendatalab/MinerU) 本地 CLI（v3.4+，GPU 自动检测） | 本地部署，支持 `hybrid-engine`（GPU）或 `pipeline`（CPU），模型自动下载                                                   |

所有输出采用统一格式：YAML 前言 → 干净的 Markdown 正文 → `images/` 子目录。

## 安装

需要 **Python >= 3.11**。

```bash
cd paper2md
pip install -e .

# 可选：本地 GPU MinerU（详见 https://github.com/opendatalab/MinerU）
pip install -e ".[local-mineru]"
# 或等效命令：
# pip install "mineru[all]>=3.4.0"
```

## MCP 配置

可以使用控制台脚本（更简洁）或模块路径：

```json
{
  "mcpServers": {
    "paper2md": {
      "command": "paper2md",
      "args": [],
      "env": {
        "MINERU_API_KEY": "sk-your-key-here",
        "MINERU_BACKEND": "hybrid-engine",
        "MODEL_VERSION": "vlm",
        "ENABLE_TABLE": "true",
        "ENABLE_FORMULA": "true"
      }
    }
  }
}
```

如果 `paper2md` 不在 PATH 中，可使用模块形式代替：

```json
{
  "mcpServers": {
    "paper2md": {
      "command": "python",
      "args": ["-m", "paper2md.server"],
      "env": {
        "MINERU_API_KEY": "sk-your-key-here",
        "MINERU_BACKEND": "hybrid-engine",
        "MODEL_VERSION": "vlm",
        "ENABLE_TABLE": "true",
        "ENABLE_FORMULA": "true"
      }
    }
  }
}
```

## 工具

### `convert_paper`

| 参数           | 类型    | 必填 | 说明                                                              |
| -------------- | ------- | ---- | ----------------------------------------------------------------- |
| `source`       | string  | ✅   | arXiv ID/URL、Nature URL/DOI、PDF URL 或本地 PDF 路径             |
| `output_dir`   | string  | ✅   | 输出目录的绝对路径                                                |
| `language`     | string  | ❌   | 文档语言：`en`、`ch`、`japan`、`korean` 等（默认：`en`）          |
| `is_ocr`       | boolean | ❌   | 对扫描版 PDF 强制 OCR（默认：`false`）                            |

### 部署时环境变量

| 变量                 | 默认值     | 说明                                                                                       |
| -------------------- | ---------- | ------------------------------------------------------------------------------------------ |
| `MINERU_API_KEY`     | *（空）*   | MinerU 云端 API token。为空时使用 Agent API（无需密钥，限制 10MB / 20 页）                  |
| `MINERU_USE_LOCAL`   | `false`    | 即使已设置 API key 也强制使用本地 GPU 部署                                                  |
| `MINERU_BACKEND`     | `auto`     | `auto`、`pipeline`（CPU）、`vlm-engine`（GPU）、`hybrid-engine`（GPU，推荐）                 |
| `MODEL_VERSION`      | `vlm`      | `vlm`（推荐）、`pipeline`、`MinerU-HTML`                                                    |
| `ENABLE_TABLE`       | `true`     | 启用表格识别                                                                               |
| `ENABLE_FORMULA`     | `true`     | 启用公式识别                                                                               |

> **注意：** 关于 MinerU 后端、模型版本、硬件要求及高级配置的完整文档，请参阅
> [MinerU 文档](https://opendatalab.github.io/MinerU/)。

### `list_supported_sources`

返回所有已注册来源提取器及其描述。用于查看支持的输入格式。

## 支持的输入格式

| 来源类型       | 示例                                                                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **arXiv**      | `2301.12345`、`https://arxiv.org/abs/2301.12345`、`https://ar5iv.labs.arxiv.org/html/2301.12345`、`https://arxiv.org/pdf/2301.12345.pdf`              |
| **Nature**     | `https://www.nature.com/articles/s41586-...`、`10.1038/s41586-...`                                                                                    |
| **PDF URL**    | `https://example.com/paper.pdf`                                                                                                                       |
| **PDF 本地**   | `/path/to/paper.pdf`、`C:\papers\paper.pdf`                                                                                                           |

## 输出结构

```
output_dir/
├── paper_title.md          # YAML 前言 + Markdown 正文
└── images/                 # 图片（内容哈希命名）
    ├── a1b2c3d4e5f6.png
    └── ...
```

### 输出示例

```markdown
---
title: "Attention Is All You Need"
authors: [Ashish Vaswani, Noam Shazeer]
doi: "10.48550/arXiv.1706.03762"
arxiv_id: "1706.03762"
year: "2017"
---

## Abstract

The dominant sequence transduction models are based on complex recurrent...

## 1 Introduction

...

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

![Figure 1: The Transformer architecture](images/a1b2c3d4e5f6.png)

| Model | BLEU | Training Cost |
| --- | --- | --- |
| Transformer | 28.4 | 3.3e18 |
```

## 扩展新来源

只需创建一个文件即可添加新论文来源——注册器会自动发现。

1. 创建 `src/paper2md/sources/pubmed.py`
2. 继承 `BaseExtractor`：

```python
from paper2md.sources.base import BaseExtractor
from paper2md.models import ConvertConfig, ParseResult

class PubMedExtractor(BaseExtractor):
    @property
    def source_type(self) -> str:
        return "pubmed"

    @property
    def display_name(self) -> str:
        return "PubMed"

    async def can_handle(self, config: ConvertConfig) -> bool:
        return "pubmed" in config.source or "pmid" in config.source

    async def extract(self, config: ConvertConfig) -> ParseResult:
        # 你的提取逻辑
        return ParseResult(
            markdown_body="# Extracted Content\n...",
            source_type="pubmed",
        )
```

就这样——无需修改其他任何文件。

## 架构

```
src/paper2md/
├── server.py              # MCP 服务端（stdio 传输）
├── models.py              # 数据模型（ConvertConfig、ParseResult 等）
├── pipeline.py            # 编排：检测 → 提取 → 写入
├── sources/               # 可扩展的来源提取器（自动发现）
│   ├── base.py            #   抽象基类 BaseExtractor
│   ├── registry.py        #   自动发现 SourceRegistry
│   ├── arxiv.py           #   arXiv：ar5iv HTML → 结构化 DOM 遍历器
│   ├── nature.py          #   Nature：语义化 HTML 解析器
│   └── pdf.py             #   PDF：URL + 本地文件（MinerU 后端）
├── mineru/                # MinerU 文档解析集成
│   ├── client.py          #   云端 API（precision v4 + agent v1）
│   └── local.py           #   本地 GPU（mineru CLI v3.4+）
└── storage/               # 输出管理
    └── __init__.py         #   Markdown 写入器 + 图片资源管理
```

## 许可证

MIT
