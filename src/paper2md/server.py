"""MCP Server entry point for paper2md.

Exposes the `convert_paper` tool via the Model Context Protocol.
Follows MCP development best practices for tool definition, error
handling, and progress reporting.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server, InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    ServerCapabilities,
)

from .models import ConvertConfig
from .pipeline import ConversionPipeline
from .sources.registry import SourceRegistry

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,  # MCP uses stderr for logging, stdout for protocol
)
logger = logging.getLogger("paper2md.server")

# ---------------------------------------------------------------------------
# Deploy-time configuration (environment variables only)
# ---------------------------------------------------------------------------
# These are read once at server startup. Per-document parameters (source,
# output_dir, language, is_ocr) come from the tool call arguments.
#
#   MINERU_API_KEY    MinerU cloud API token (empty = use agent API, no key)
#   MINERU_USE_LOCAL  Force local GPU deployment (true / false)
#   MINERU_BACKEND    Backend: auto / pipeline (CPU) / hybrid-engine (GPU)
#   MODEL_VERSION     Model: vlm (recommended) / pipeline / MinerU-HTML
#   ENABLE_TABLE      Enable table recognition (true / false)
#   ENABLE_FORMULA    Enable formula recognition (true / false)
# ---------------------------------------------------------------------------

MINERU_API_KEY = os.environ.get("MINERU_API_KEY", "")
MINERU_USE_LOCAL = os.environ.get("MINERU_USE_LOCAL", "").lower() in (
    "1", "true", "yes", "on"
)
MINERU_BACKEND = os.environ.get("MINERU_BACKEND", "auto")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "vlm")
ENABLE_TABLE = os.environ.get("ENABLE_TABLE", "true").lower() not in (
    "0", "false", "no", "off"
)
ENABLE_FORMULA = os.environ.get("ENABLE_FORMULA", "true").lower() not in (
    "0", "false", "no", "off"
)

logger.info(
    "MinerU config: use_local=%s, backend=%s, model=%s, api_key=%s, table=%s, formula=%s",
    MINERU_USE_LOCAL, MINERU_BACKEND, MODEL_VERSION,
    "***" if MINERU_API_KEY else "(not set)",
    ENABLE_TABLE, ENABLE_FORMULA,
)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server = Server("paper2md")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

CONVERT_PAPER_TOOL = Tool(
    name="convert_paper",
    description=(
        "A useful tool to obtain a clean and full Markdown version of an academic paper from various sources. "
        "Convert an academic paper to clean Markdown with YAML metadata, "
        "LaTeX formulas, tables, and locally saved images."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Paper link or local file path.",
            },
            "output_dir": {
                "type": "string",
                "description": "Output directory for the markdown file and images/ folder.",
            },
            "language": {
                "type": "string",
                "description": "Document language (default: 'en').",
                "default": "en",
            },
            "is_ocr": {
                "type": "boolean",
                "description": "Force OCR for scanned PDFs (default: false).",
                "default": False,
            },
        },
        "required": ["source", "output_dir"],
    },
)

LIST_SOURCES_TOOL = Tool(
    name="list_supported_sources",
    description=(
        "List all supported paper sources and their descriptions. "
        "Useful for discovering what input formats are accepted."
    ),
    inputSchema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)

# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """Return the list of available tools."""
    return [CONVERT_PAPER_TOOL, LIST_SOURCES_TOOL]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Route tool calls to the appropriate handler."""
    try:
        if name == "convert_paper":
            return await _handle_convert_paper(arguments)
        elif name == "list_supported_sources":
            return await _handle_list_sources()
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}",
            )]
    except ValueError as e:
        logger.warning("Validation error: %s", e)
        return [TextContent(
            type="text",
            text=f"Error: {e}",
        )]
    except Exception as e:
        logger.exception("Unexpected error in tool '%s'", name)
        return [TextContent(
            type="text",
            text=f"Unexpected error: {e}",
        )]


async def _handle_convert_paper(
    arguments: dict[str, Any]
) -> list[TextContent]:
    """Handle the convert_paper tool call."""
    source = arguments.get("source", "").strip()
    output_dir_str = arguments.get("output_dir", "").strip()

    if not source:
        return [TextContent(
            type="text",
            text="Error: 'source' parameter is required.",
        )]
    if not output_dir_str:
        return [TextContent(
            type="text",
            text="Error: 'output_dir' parameter is required.",
        )]

    output_dir = Path(output_dir_str)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    # Build config from arguments + server environment
    config = ConvertConfig(
        source=source,
        output_dir=output_dir,
        mineru_api_key=MINERU_API_KEY or None,
        mineru_use_local=MINERU_USE_LOCAL,
        mineru_backend=MINERU_BACKEND,
        model_version=MODEL_VERSION,
        language=arguments.get("language", "en"),
        is_ocr=arguments.get("is_ocr", False),
        enable_table=ENABLE_TABLE,
        enable_formula=ENABLE_FORMULA,
    )

    logger.info("Converting paper: source=%s, output=%s", source, output_dir)

    # Run pipeline
    pipeline = ConversionPipeline()
    try:
        md_path = await pipeline.run(config)

        # Read the result for summary
        content = md_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        char_count = len(content)
        image_dir = output_dir / "images"
        image_count = (
            len(list(image_dir.glob("*"))) if image_dir.exists() else 0
        )

        summary = (
            f"✅ Conversion successful!\n\n"
            f"**Source**: {source}\n"
            f"**Output**: {md_path}\n"
            f"**Size**: {char_count:,} characters, {len(lines)} lines\n"
            f"**Images**: {image_count} saved to {image_dir}\n\n"
            f"First 50 lines of output:\n```markdown\n"
            f"{chr(10).join(lines[:50])}\n```"
        )

        return [TextContent(type="text", text=summary)]

    except ValueError as e:
        # Source not recognized — provide helpful error
        sources = SourceRegistry.list_sources()
        source_list = "\n".join(
            f"  • **{s['type']}** ({s['name']}): {s.get('description', '')[:100]}"
            for s in sources
        )
        return [TextContent(
            type="text",
            text=(
                f"❌ Could not recognize the paper source.\n\n"
                f"Source provided: `{source}`\n"
                f"Error: {e}\n\n"
                f"**Supported sources**:\n{source_list}\n\n"
                f"Try providing a full URL or checking the format."
            ),
        )]
    except RuntimeError as e:
        return [TextContent(
            type="text",
            text=f"❌ Conversion failed: {e}",
        )]


async def _handle_list_sources() -> list[TextContent]:
    """Handle the list_supported_sources tool call."""
    sources = SourceRegistry.list_sources()

    if not sources:
        return [TextContent(
            type="text",
            text="No sources registered. Check the installation.",
        )]

    lines = ["## Supported Paper Sources\n"]
    for s in sources:
        lines.append(f"### {s['name']} (`{s['type']}`)")
        if s.get("description"):
            desc = s["description"].strip().split("\n")[0]
            lines.append(f"{desc}\n")

    return [TextContent(type="text", text="\n".join(lines))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the paper2md MCP server over stdio (standard MCP transport)."""
    logger.info("Starting paper2md MCP server v1.0.0")

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="paper2md",
                    server_version="1.0.0",
                    capabilities=ServerCapabilities(tools={}),
                ),
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception:
        logger.exception("Server crashed")
        raise


if __name__ == "__main__":
    main()
