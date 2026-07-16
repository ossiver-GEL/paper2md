"""Comprehensive test script for paper2md — covers all pipeline paths.

Test cases:
  1. arXiv via ar5iv HTML (primary path) — "Attention Is All You Need" 1706.03762
  2. arXiv via ar5iv (ML paper) — "Graph ML" 2302.07459
  3. arXiv via ar5iv (math-heavy) — "Transformer survey" or math paper
  4. arXiv fallback (broken ar5iv) — "Adam" 1412.6980
  5. Nature article
  6. PDF URL via MinerU cloud
  7. list_supported_sources tool
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from paper2md.models import ConvertConfig
from paper2md.pipeline import ConversionPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)

TEST_OUTPUT = Path(__file__).parent / "test_output"

RESULTS: list[dict] = []


def report(case: str, status: str, detail: str = "") -> None:
    icon = "[OK]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
    print(f"\n{'='*60}")
    print(f"{icon} {case}: {status}")
    if detail:
        print(f"   {detail}")
    RESULTS.append({"case": case, "status": status, "detail": detail})


# ---------------------------------------------------------------------------
# Test: list_supported_sources
# ---------------------------------------------------------------------------
async def test_list_sources() -> None:
    case = "list_supported_sources"
    try:
        pipeline = ConversionPipeline()
        sources = pipeline.list_sources()
        names = [s["type"] for s in sources]
        expected = {"arxiv", "nature", "pdf_url", "pdf_local"}
        if expected.issubset(set(names)):
            report(case, "PASS", f"Found: {', '.join(names)}")
        else:
            report(case, "FAIL", f"Missing: {expected - set(names)}. Got: {names}")
    except Exception as e:
        report(case, "FAIL", str(e))


# ---------------------------------------------------------------------------
# Test: arXiv via ar5iv (primary path)
# ---------------------------------------------------------------------------
async def test_arxiv_ar5iv(arxiv_id: str, label: str) -> None:
    case = f"arXiv ar5iv — {label}"
    try:
        config = ConvertConfig(
            source=arxiv_id,
            output_dir=TEST_OUTPUT / f"arxiv_{arxiv_id.replace('/', '_')}",
        )
        pipeline = ConversionPipeline()
        md_path = await pipeline.run(config)

        content = md_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        headings = [l for l in lines if l.startswith("## ")]

        # Quality checks
        has_frontmatter = content.startswith("---")
        has_abstract = any("abstract" in l.lower() and l.startswith("##") for l in lines)
        char_count = len(content)

        if has_frontmatter and has_abstract and char_count > 500:
            report(
                case, "PASS",
                f"{char_count:,} chars, {len(lines)} lines, "
                f"{len(headings)} sections, "
                f"output: {md_path.relative_to(TEST_OUTPUT)}",
            )
        else:
            issues = []
            if not has_frontmatter:
                issues.append("no frontmatter")
            if not has_abstract:
                issues.append("no abstract")
            if char_count <= 500:
                issues.append(f"too short ({char_count} chars)")
            report(case, "FAIL", "; ".join(issues))
    except Exception as e:
        report(case, "FAIL", str(e))


# ---------------------------------------------------------------------------
# Test: arXiv fallback (broken ar5iv → abstract page)
# ---------------------------------------------------------------------------
async def test_arxiv_fallback(arxiv_id: str, label: str) -> None:
    case = f"arXiv fallback — {label}"
    try:
        config = ConvertConfig(
            source=arxiv_id,
            output_dir=TEST_OUTPUT / f"arxiv_fb_{arxiv_id.replace('/', '_')}",
        )
        pipeline = ConversionPipeline()
        md_path = await pipeline.run(config)

        content = md_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        char_count = len(content)

        # Fallback should produce abstract-only (shorter but valid)
        has_frontmatter = content.startswith("---")
        if has_frontmatter and char_count > 200:
            report(
                case, "PASS",
                f"{char_count:,} chars (abstract-only fallback), "
                f"output: {md_path.relative_to(TEST_OUTPUT)}",
            )
        else:
            report(case, "FAIL", f"Too short ({char_count} chars)")
    except Exception as e:
        report(case, "FAIL", str(e))


# ---------------------------------------------------------------------------
# Test: Nature article
# ---------------------------------------------------------------------------
async def test_nature(
    url_or_doi: str,
    label: str,
    expect_cc: bool = False,
) -> None:
    case = f"Nature — {label}"
    try:
        config = ConvertConfig(
            source=url_or_doi,
            output_dir=TEST_OUTPUT / "nature_test",
        )
        pipeline = ConversionPipeline()
        md_path = await pipeline.run(config)

        content = md_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        headings = [l for l in lines if l.startswith("## ")]
        char_count = len(content)

        has_frontmatter = content.startswith("---")
        has_cc = "CC" in content or "Creative Commons" in content or "cc" in content.lower()

        if has_frontmatter and char_count > 500 and len(headings) >= 2:
            cc_note = " (CC license detected)" if has_cc else ""
            report(
                case, "PASS",
                f"{char_count:,} chars, {len(lines)} lines, "
                f"{len(headings)} sections{cc_note}, "
                f"output: {md_path.relative_to(TEST_OUTPUT)}",
            )
        else:
            issues = []
            if not has_frontmatter:
                issues.append("no frontmatter")
            if char_count <= 500:
                issues.append(f"too short ({char_count} chars)")
            if len(headings) < 2:
                issues.append(f"only {len(headings)} sections")
            report(case, "FAIL", "; ".join(issues))
    except Exception as e:
        report(case, "FAIL", str(e))


# ---------------------------------------------------------------------------
# Test: PDF URL via MinerU cloud
# ---------------------------------------------------------------------------
async def test_pdf_url(pdf_url: str, label: str) -> None:
    case = f"PDF URL — {label}"
    try:
        config = ConvertConfig(
            source=pdf_url,
            output_dir=TEST_OUTPUT / "pdf_url_test",
        )
        pipeline = ConversionPipeline()
        md_path = await pipeline.run(config)

        content = md_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        char_count = len(content)

        if char_count > 100:
            report(
                case, "PASS",
                f"{char_count:,} chars, {len(lines)} lines, "
                f"output: {md_path.relative_to(TEST_OUTPUT)}",
            )
        else:
            report(case, "FAIL", f"Too short ({char_count} chars)")
    except Exception as e:
        report(case, "FAIL", str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    print("=" * 60)
    print("paper2md — Comprehensive Pipeline Test")
    print("=" * 60)

    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)

    # 1. List sources
    await test_list_sources()

    # 2. arXiv ar5iv (good papers — should use ar5iv path)
    await test_arxiv_ar5iv("1706.03762", "Attention Is All You Need")  # well-known, works
    await test_arxiv_ar5iv("2302.07459", "Graph ML Survey")           # moderate complexity
    await test_arxiv_ar5iv("1812.04948", "StyleGAN")                  # figures-heavy

    # 3. arXiv fallback (broken ar5iv → abstract page)
    await test_arxiv_fallback("1412.6980", "Adam optimizer")           # known-broken ar5iv

    # 4. Nature article
    await test_nature(
        "https://www.nature.com/articles/s41586-023-06221-2",
        "AlphaFold-related",
    )

    # 5. PDF URL
    await test_pdf_url(
        "https://arxiv.org/pdf/1706.03762.pdf",
        "Attention Is All You Need (PDF)",
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    for r in RESULTS:
        icon = "[OK]" if r["status"] == "PASS" else "[FAIL]" if r["status"] == "FAIL" else "[WARN]"
        print(f"  {icon} {r['case']}")
    print(f"\n{passed}/{len(RESULTS)} passed, {failed} failed")


if __name__ == "__main__":
    asyncio.run(main())
