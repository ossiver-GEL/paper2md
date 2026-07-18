# paper2md — Academic Paper to Markdown MCP Tool

A Model Context Protocol (MCP) tool that converts academic papers into clean,
structured Markdown with YAML frontmatter and local images. Supports multiple
paper sources with source-specific extraction for maximum fidelity.

## How It Works

paper2md uses a **tiered extraction strategy** optimized for each source:

| Source                 | Primary Method                              | Reason                                                                                                           |
| ---------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **arXiv**        | ar5iv HTML → structural DOM walker         | ar5iv = LaTeXML output (same engine as arXiv's official HTML). Already the best LaTeX→HTML converter available. |
| **Nature**       | Nature.com HTML → semantic section parser  | Nature provides clean`<section data-title>` markup. Direct parsing = zero HTML artifacts.                      |
| **PDF (remote)** | [MinerU](https://github.com/opendatalab/MinerU) cloud API (precision v4 or agent v1) | SOTA document parsing for PDF. Precision API with token, agent API without.                                      |
| **PDF (local)**  | [MinerU](https://github.com/opendatalab/MinerU) local CLI (v3.4+, GPU auto-detect)   | Local deployment via`hybrid-engine` (GPU) or `pipeline` (CPU). Models auto-download.                         |

All outputs share the same format: YAML frontmatter → clean Markdown body → `images/` subdirectory.

## Installation

Requires **Python >= 3.11**.

```bash
cd paper2md
pip install -e .

# Optional: local GPU MinerU (see https://github.com/opendatalab/MinerU)
pip install -e ".[local-mineru]"
# or equivalently:
# pip install "mineru[all]>=3.4.0"
```

## MCP Configuration

Use either the console script (simpler) or the module path:

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

If `paper2md` is not on PATH, use the module form instead:

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

## Tools

### `convert_paper`

| Parameter      | Type    | Required | Description                                                                     |
| -------------- | ------- | -------- | ------------------------------------------------------------------------------- |
| `source`     | string  | ✅       | arXiv ID/URL, Nature URL/DOI, DOI, PMID, PMCID, PubMed/PMC URL, PDF URL, or local PDF path                        |
| `output_dir` | string  | ✅       | Absolute path to output directory                                               |
| `language`   | string  | ❌       | Document language:`en`, `ch`, `japan`, `korean`, etc. (default: `en`) |
| `is_ocr`     | boolean | ❌       | Force OCR for scanned PDFs (default:`false`)                                  |

### Deploy-Time Environment Variables

| Variable             | Default     | Description                                                                              |
| -------------------- | ----------- | ---------------------------------------------------------------------------------------- |
| `MINERU_API_KEY`   | *(empty)* | MinerU cloud API token. Empty = use agent API (no key, 10MB/20 page limit)               |
| `MINERU_USE_LOCAL` | `false`   | Force local GPU deployment even if API key is set                                        |
| `MINERU_BACKEND`   | `auto`    | `auto`, `pipeline` (CPU), `vlm-engine` (GPU), `hybrid-engine` (GPU, recommended) |
| `MODEL_VERSION`    | `vlm`     | `vlm` (recommended), `pipeline`, `MinerU-HTML`                                     |
| `ENABLE_TABLE`     | `true`    | Enable table recognition                                                                 |
| `ENABLE_FORMULA`   | `true`    | Enable formula recognition                                                               |

> **Note:** For full documentation on MinerU backends, model versions, hardware
> requirements, and advanced configuration, see the [MinerU Documentation](https://opendatalab.github.io/MinerU/).

### `list_supported_sources`

Returns all registered source extractors with descriptions. Useful for discovering
what input formats are accepted.

## Supported Input Formats

| Source Type         | Examples                                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **arXiv**     | `2301.12345`, `https://arxiv.org/abs/2301.12345`, `https://ar5iv.labs.arxiv.org/html/2301.12345`, `https://arxiv.org/pdf/2301.12345.pdf` |
| **Nature**    | `https://www.nature.com/articles/s41586-...`, `10.1038/s41586-...`                                                                           |
| **Europe PMC** | `10.1016/j.eclinm.2022.101704`, `PMID:36467456`, `PMC9716327`, `https://pubmed.ncbi.nlm.nih.gov/36467456`, `https://europepmc.org/article/MED/36467456` |
| **PDF URL**   | `https://example.com/paper.pdf`                                                                                                                |
| **PDF Local** | `/path/to/paper.pdf`, `C:\papers\paper.pdf`                                                                                                  |

## Output Structure

```
output_dir/
├── paper_title.md          # YAML frontmatter + Markdown body
└── images/                 # Figures (content-hash filenames)
    ├── a1b2c3d4e5f6.png
    └── ...
```

### Example Output

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

## Extending with New Sources

Add a new paper source by creating a single file — the registry discovers it automatically.

1. Create `src/paper2md/sources/pubmed.py`
2. Subclass `BaseExtractor`:

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
        # Your extraction logic
        return ParseResult(
            markdown_body="# Extracted Content\n...",
            source_type="pubmed",
        )
```

That's it — no other files need modification.

## Architecture

```
src/paper2md/
├── server.py              # MCP server (stdio transport)
├── models.py              # Data models (ConvertConfig, ParseResult, etc.)
├── pipeline.py            # Orchestration: detect → extract → write
├── sources/               # Extensible source extractors (auto-discovered)
│   ├── base.py            #   Abstract BaseExtractor
│   ├── registry.py        #   Auto-discovery SourceRegistry
│   ├── arxiv.py           #   arXiv: ar5iv HTML → structural DOM walker
│   ├── nature.py          #   Nature: semantic HTML parser
│   ├── europe_pmc.py      #   Europe PMC: JATS XML → Markdown (OA papers in PMC)
│   ├── jats_parser.py     #   Reusable JATS XML → Markdown converter
│   └── pdf.py             #   PDF: URL + local file (MinerU backend)
├── mineru/                # MinerU document parsing integration
│   ├── client.py          #   Cloud API (precision v4 + agent v1)
│   └── local.py           #   Local GPU (mineru CLI v3.4+)
└── storage/               # Output management
    └── __init__.py         #   Markdown writer + image asset management
```

## License

MIT
