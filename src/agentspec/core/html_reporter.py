"""Self-contained HTML report generator for agentspec test results.

Produces a single HTML file with embedded CSS and JavaScript -- no external
dependencies, no CDN links.  The report is professional enough to share with
teammates or attach to a pull request.

Usage::

    from agentspec.core.html_reporter import HTMLReporter

    reporter = HTMLReporter()
    html = reporter.generate(suite, fix_report=fix_report)
    reporter.save(suite, "report.html", fix_report=fix_report)
"""

from __future__ import annotations

import html
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentspec.models.test import (
    TestCategory,
    TestResult,
    TestSeverity,
    TestStatus,
    TestSuite,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATUS_ICONS: dict[TestStatus, str] = {
    TestStatus.PASSED: "&#10003;",   # checkmark
    TestStatus.FAILED: "&#10007;",   # ballot X
    TestStatus.ERROR: "&#9888;",     # warning sign
    TestStatus.SKIPPED: "&#9675;",   # white circle
}

_STATUS_LABELS: dict[TestStatus, str] = {
    TestStatus.PASSED: "Passed",
    TestStatus.FAILED: "Failed",
    TestStatus.ERROR: "Error",
    TestStatus.SKIPPED: "Skipped",
}

_SEVERITY_LABELS: dict[TestSeverity, str] = {
    TestSeverity.CRITICAL: "Critical",
    TestSeverity.HIGH: "High",
    TestSeverity.MEDIUM: "Medium",
    TestSeverity.LOW: "Low",
    TestSeverity.INFO: "Info",
}

_CATEGORY_LABELS: dict[TestCategory, str] = {
    TestCategory.HAPPY_PATH: "Happy Path",
    TestCategory.EDGE_CASE: "Edge Case",
    TestCategory.ERROR_HANDLING: "Error Handling",
    TestCategory.SECURITY: "Security",
    TestCategory.SCHEMA_VALIDATION: "Schema Validation",
    TestCategory.PERFORMANCE: "Performance",
}

# Deterministic ordering for categories.
_CATEGORY_ORDER: list[TestCategory] = [
    TestCategory.HAPPY_PATH,
    TestCategory.EDGE_CASE,
    TestCategory.ERROR_HANDLING,
    TestCategory.SECURITY,
    TestCategory.SCHEMA_VALIDATION,
    TestCategory.PERFORMANCE,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    """HTML-escape a value, coercing to string first."""
    return html.escape(str(value))


def _json_preview(body: Any, max_length: int = 500) -> str:
    """Produce a truncated, pretty-printed JSON preview of *body*."""
    if body is None:
        return ""
    try:
        text = json.dumps(body, indent=2, default=str)
    except (TypeError, ValueError):
        text = str(body)
    if len(text) > max_length:
        text = text[:max_length] + "\n..."
    return text


def _format_duration(ms: float) -> str:
    """Format a duration in milliseconds to a human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


def _severity_css_class(severity: TestSeverity) -> str:
    """Map a severity to a CSS class name."""
    return f"sev-{severity.value}"


def _status_css_class(status: TestStatus) -> str:
    """Map a status to a CSS class name."""
    return f"st-{status.value}"


# ---------------------------------------------------------------------------
# SVG generators
# ---------------------------------------------------------------------------


def _donut_svg(pass_rate: float, size: int = 120) -> str:
    """Generate an inline SVG donut chart for the pass rate.

    Parameters
    ----------
    pass_rate:
        A percentage value between 0 and 100.
    size:
        Width and height of the SVG viewport in pixels.
    """
    radius = 45
    circumference = 2 * 3.14159265 * radius
    filled = circumference * (pass_rate / 100.0)
    empty = circumference - filled

    if pass_rate >= 90:
        colour = "var(--color-pass)"
    elif pass_rate >= 70:
        colour = "var(--color-warn)"
    else:
        colour = "var(--color-fail)"

    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
        f'class="donut-chart">'
        f'<circle cx="{size // 2}" cy="{size // 2}" r="{radius}" '
        f'fill="none" stroke="var(--bg-tertiary)" stroke-width="10" />'
        f'<circle cx="{size // 2}" cy="{size // 2}" r="{radius}" '
        f'fill="none" stroke="{colour}" stroke-width="10" '
        f'stroke-dasharray="{filled:.2f} {empty:.2f}" '
        f'stroke-dashoffset="{circumference * 0.25:.2f}" '
        f'stroke-linecap="round" '
        f'style="transition: stroke-dasharray 1s ease;" />'
        f'<text x="50%" y="50%" text-anchor="middle" '
        f'dominant-baseline="central" '
        f'fill="{colour}" font-size="22" font-weight="700" '
        f'font-family="var(--font-mono)">'
        f'{pass_rate:.1f}%</text>'
        f'</svg>'
    )


def _category_bar_chart_svg(
    by_category: dict[TestCategory, list[TestResult]],
) -> str:
    """Generate a horizontal bar chart SVG showing pass/fail per category."""
    if not by_category:
        return ""

    bar_height = 28
    gap = 6
    label_width = 140
    chart_width = 500
    total_height = len(by_category) * (bar_height + gap) + gap

    bars: list[str] = []
    y = gap

    for category in _CATEGORY_ORDER:
        results = by_category.get(category)
        if not results:
            continue

        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = total - passed
        pass_width = (passed / total * (chart_width - label_width)) if total else 0
        fail_width = (failed / total * (chart_width - label_width)) if total else 0
        label = _CATEGORY_LABELS.get(category, category.value)

        bars.append(
            f'<text x="0" y="{y + bar_height * 0.7}" '
            f'fill="var(--text-primary)" font-size="12" '
            f'font-family="var(--font-sans)">{_esc(label)}</text>'
        )
        # Passed bar
        if pass_width > 0:
            bars.append(
                f'<rect x="{label_width}" y="{y}" '
                f'width="{pass_width:.1f}" height="{bar_height}" '
                f'rx="3" fill="var(--color-pass)" opacity="0.85" />'
            )
        # Failed bar
        if fail_width > 0:
            bars.append(
                f'<rect x="{label_width + pass_width:.1f}" y="{y}" '
                f'width="{fail_width:.1f}" height="{bar_height}" '
                f'rx="3" fill="var(--color-fail)" opacity="0.85" />'
            )
        # Count label
        bars.append(
            f'<text x="{label_width + pass_width + fail_width + 8:.1f}" '
            f'y="{y + bar_height * 0.7}" fill="var(--text-secondary)" '
            f'font-size="11" font-family="var(--font-mono)">'
            f'{passed}/{total}</text>'
        )
        y += bar_height + gap

    bar_content = "\n".join(bars)
    return (
        f'<svg viewBox="0 0 {chart_width + 60} {total_height}" '
        f'width="100%" height="{total_height}" '
        f'class="category-chart" preserveAspectRatio="xMinYMin meet">'
        f'{bar_content}'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_css() -> str:
    """Return the full embedded CSS stylesheet."""
    return """\
:root {
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
  --font-mono: "SF Mono", "Fira Code", "Fira Mono", "Roboto Mono", Consolas, monospace;

  /* Light theme */
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-tertiary: #e2e8f0;
  --bg-code: #f1f5f9;
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #94a3b8;
  --border-color: #e2e8f0;
  --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
  --shadow-lg: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);

  --color-pass: #22c55e;
  --color-fail: #ef4444;
  --color-warn: #eab308;
  --color-info: #3b82f6;
  --color-skip: #94a3b8;
  --color-accent: #6366f1;
}

[data-theme="dark"] {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-tertiary: #334155;
  --bg-code: #1e293b;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --border-color: #334155;
  --shadow: 0 1px 3px rgba(0,0,0,0.3);
  --shadow-lg: 0 4px 6px rgba(0,0,0,0.35);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-sans);
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.container { max-width: 1200px; margin: 0 auto; padding: 24px; }

/* --- Header --- */
.report-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 32px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}

.header-info { flex: 1; }

.logo {
  font-size: 28px;
  font-weight: 800;
  color: var(--color-accent);
  letter-spacing: -0.5px;
}

.api-name {
  font-size: 20px;
  font-weight: 600;
  margin-top: 8px;
  color: var(--text-primary);
}

.api-source {
  font-size: 13px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
  margin-top: 4px;
  word-break: break-all;
}

.timestamp {
  font-size: 13px;
  color: var(--text-muted);
  margin-top: 8px;
}

.header-chart { text-align: center; flex-shrink: 0; margin-left: 32px; }

.theme-toggle {
  position: fixed;
  top: 16px;
  right: 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 8px 12px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  z-index: 100;
  box-shadow: var(--shadow);
  transition: background 0.2s;
}
.theme-toggle:hover { background: var(--bg-tertiary); }

/* --- Summary cards --- */
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.summary-card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 10px;
  padding: 20px;
  text-align: center;
  box-shadow: var(--shadow);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.summary-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-lg); }

.card-value {
  font-size: 32px;
  font-weight: 700;
  font-family: var(--font-mono);
  line-height: 1.2;
}
.card-label {
  font-size: 13px;
  color: var(--text-secondary);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.card-pass .card-value { color: var(--color-pass); }
.card-fail .card-value { color: var(--color-fail); }
.card-error .card-value { color: var(--color-warn); }
.card-skip .card-value { color: var(--color-skip); }
.card-critical .card-value { color: var(--color-fail); }
.card-duration .card-value { color: var(--color-info); }
.card-total .card-value { color: var(--color-accent); }

.critical-alert {
  background: #fef2f2;
  border-color: #fecaca;
}
[data-theme="dark"] .critical-alert {
  background: #450a0a;
  border-color: #7f1d1d;
}

/* --- Section --- */
.section {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
  overflow: hidden;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-primary);
}

.section-body { padding: 20px; }

/* --- Filters --- */
.filters {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  padding: 12px 20px;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-primary);
  align-items: center;
}

.filter-group { display: flex; gap: 4px; align-items: center; }

.filter-label {
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-right: 4px;
}

.filter-btn {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  padding: 4px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}
.filter-btn:hover { background: var(--bg-tertiary); color: var(--text-primary); }
.filter-btn.active {
  background: var(--color-accent);
  border-color: var(--color-accent);
  color: #fff;
}

/* --- Results table --- */
.results-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.results-table th {
  text-align: left;
  padding: 10px 12px;
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  border-bottom: 2px solid var(--border-color);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
.results-table th:hover { color: var(--text-primary); }
.results-table th .sort-arrow { margin-left: 4px; font-size: 10px; }

.results-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border-color);
  vertical-align: middle;
}

.results-table tbody tr {
  transition: background 0.1s;
  cursor: pointer;
}
.results-table tbody tr:hover { background: var(--bg-tertiary); }

.status-icon { font-size: 16px; font-weight: 700; }
.st-passed .status-icon { color: var(--color-pass); }
.st-failed .status-icon { color: var(--color-fail); }
.st-error .status-icon { color: var(--color-warn); }
.st-skipped .status-icon { color: var(--color-skip); }

/* --- Badges --- */
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.badge-category {
  background: var(--bg-tertiary);
  color: var(--text-secondary);
}

.sev-critical .badge-severity { background: #fef2f2; color: #dc2626; }
.sev-high .badge-severity    { background: #fff7ed; color: #ea580c; }
.sev-medium .badge-severity  { background: #fefce8; color: #ca8a04; }
.sev-low .badge-severity     { background: #f0f9ff; color: #0284c7; }
.sev-info .badge-severity    { background: #f8fafc; color: #64748b; }

[data-theme="dark"] .sev-critical .badge-severity { background: #450a0a; color: #fca5a5; }
[data-theme="dark"] .sev-high .badge-severity    { background: #431407; color: #fdba74; }
[data-theme="dark"] .sev-medium .badge-severity  { background: #422006; color: #fde047; }
[data-theme="dark"] .sev-low .badge-severity     { background: #0c4a6e; color: #7dd3fc; }
[data-theme="dark"] .sev-info .badge-severity    { background: #1e293b; color: #94a3b8; }

/* --- Expandable detail row --- */
.detail-row { display: none; }
.detail-row.open { display: table-row; }

.detail-cell {
  padding: 16px 24px !important;
  background: var(--bg-primary);
  border-bottom: 2px solid var(--border-color);
}

.detail-section { margin-bottom: 12px; }

.detail-section-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  margin-bottom: 6px;
}

.failure-item {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px 14px;
  margin-bottom: 6px;
  font-size: 13px;
}

.failure-assertion { font-weight: 600; color: var(--text-primary); }
.failure-expected { color: var(--color-pass); }
.failure-actual { color: var(--color-fail); }

.response-preview {
  background: var(--bg-code);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 12px;
  font-family: var(--font-mono);
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 300px;
  overflow-y: auto;
  color: var(--text-secondary);
}

.error-msg {
  color: var(--color-fail);
  font-family: var(--font-mono);
  font-size: 13px;
  padding: 8px 12px;
  background: #fef2f2;
  border-radius: 6px;
}
[data-theme="dark"] .error-msg { background: #450a0a; }

/* --- Fix suggestions --- */
.fix-group-title {
  font-size: 14px;
  font-weight: 600;
  padding: 12px 0 8px;
  color: var(--text-primary);
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.fix-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  margin-bottom: 12px;
  overflow: hidden;
  transition: box-shadow 0.15s;
}
.fix-card:hover { box-shadow: var(--shadow-lg); }

.fix-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
}
.fix-card-header:hover { background: var(--bg-tertiary); }

.fix-card-title { font-weight: 600; font-size: 14px; }
.fix-card-toggle { color: var(--text-muted); font-size: 18px; transition: transform 0.2s; }
.fix-card.open .fix-card-toggle { transform: rotate(180deg); }

.fix-card-body {
  display: none;
  padding: 16px;
  border-top: 1px solid var(--border-color);
}
.fix-card.open .fix-card-body { display: block; }

.fix-description {
  font-size: 14px;
  color: var(--text-secondary);
  margin-bottom: 12px;
  line-height: 1.5;
}

.code-block-wrapper { position: relative; margin-bottom: 12px; }

.code-block {
  background: var(--bg-code);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 14px;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre;
  color: var(--text-primary);
}

.copy-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  color: var(--text-secondary);
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
  transition: all 0.15s;
}
.copy-btn:hover { background: var(--color-accent); color: #fff; border-color: var(--color-accent); }

.agent-instruction {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 13px;
  color: #1e40af;
  line-height: 1.5;
}
[data-theme="dark"] .agent-instruction {
  background: #172554;
  border-color: #1e3a5f;
  color: #93c5fd;
}

/* --- Utilities --- */
.text-mono { font-family: var(--font-mono); }
.text-muted { color: var(--text-muted); }
.mt-2 { margin-top: 8px; }

@media (max-width: 768px) {
  .report-header { flex-direction: column; }
  .header-chart { margin-left: 0; margin-top: 20px; }
  .summary-grid { grid-template-columns: repeat(2, 1fr); }
  .filters { flex-direction: column; }
}
"""


def _build_js() -> str:
    """Return the inline JavaScript for interactive behaviour."""
    return """\
(function() {
  "use strict";

  // --- Count-up animation ---
  function animateCountUp(el) {
    var target = parseInt(el.getAttribute("data-target"), 10);
    if (isNaN(target)) return;
    var duration = 800;
    var start = 0;
    var startTime = null;
    function step(ts) {
      if (!startTime) startTime = ts;
      var progress = Math.min((ts - startTime) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(eased * target);
      if (progress < 1) requestAnimationFrame(step);
      else el.textContent = target;
    }
    requestAnimationFrame(step);
  }

  document.querySelectorAll(".count-up").forEach(function(el) {
    animateCountUp(el);
  });

  // Animate pass rate separately (float)
  document.querySelectorAll(".count-up-float").forEach(function(el) {
    var target = parseFloat(el.getAttribute("data-target"));
    if (isNaN(target)) return;
    var duration = 800;
    var startTime = null;
    function step(ts) {
      if (!startTime) startTime = ts;
      var progress = Math.min((ts - startTime) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = (eased * target).toFixed(1) + "%";
      if (progress < 1) requestAnimationFrame(step);
      else el.textContent = target.toFixed(1) + "%";
    }
    requestAnimationFrame(step);
  });

  // --- Dark / light mode toggle ---
  var toggle = document.getElementById("themeToggle");
  if (toggle) {
    toggle.addEventListener("click", function() {
      var html = document.documentElement;
      var current = html.getAttribute("data-theme");
      var next = current === "dark" ? "light" : "dark";
      html.setAttribute("data-theme", next);
      toggle.textContent = next === "dark" ? "Light Mode" : "Dark Mode";
    });
  }

  // --- Expand / collapse result rows ---
  document.querySelectorAll(".result-row").forEach(function(row) {
    row.addEventListener("click", function() {
      var detailId = row.getAttribute("data-detail");
      var detail = document.getElementById(detailId);
      if (detail) {
        detail.classList.toggle("open");
      }
    });
  });

  // --- Expand / collapse fix cards ---
  document.querySelectorAll(".fix-card-header").forEach(function(header) {
    header.addEventListener("click", function() {
      header.parentElement.classList.toggle("open");
    });
  });

  // --- Copy to clipboard ---
  document.querySelectorAll(".copy-btn").forEach(function(btn) {
    btn.addEventListener("click", function(e) {
      e.stopPropagation();
      var codeId = btn.getAttribute("data-code");
      var codeEl = document.getElementById(codeId);
      if (codeEl) {
        navigator.clipboard.writeText(codeEl.textContent).then(function() {
          var orig = btn.textContent;
          btn.textContent = "Copied!";
          setTimeout(function() { btn.textContent = orig; }, 1500);
        });
      }
    });
  });

  // --- Filtering ---
  var activeFilters = { category: "all", status: "all", severity: "all" };

  function applyFilters() {
    document.querySelectorAll(".result-row").forEach(function(row) {
      var cat = row.getAttribute("data-category");
      var st = row.getAttribute("data-status");
      var sev = row.getAttribute("data-severity");
      var show = (activeFilters.category === "all" || activeFilters.category === cat)
              && (activeFilters.status === "all" || activeFilters.status === st)
              && (activeFilters.severity === "all" || activeFilters.severity === sev);
      row.style.display = show ? "" : "none";
      // Also hide corresponding detail row
      var detailId = row.getAttribute("data-detail");
      var detail = document.getElementById(detailId);
      if (detail) {
        if (!show) {
          detail.style.display = "none";
          detail.classList.remove("open");
        } else {
          detail.style.display = "";
        }
      }
    });
  }

  document.querySelectorAll(".filter-btn").forEach(function(btn) {
    btn.addEventListener("click", function() {
      var group = btn.getAttribute("data-filter-group");
      var value = btn.getAttribute("data-filter-value");
      // Toggle active state within the group
      btn.parentElement.querySelectorAll(".filter-btn").forEach(function(b) {
        b.classList.remove("active");
      });
      btn.classList.add("active");
      activeFilters[group] = value;
      applyFilters();
    });
  });

  // --- Sorting ---
  var sortState = { column: null, asc: true };

  function sortTable(column) {
    var table = document.getElementById("resultsTable");
    if (!table) return;
    var tbody = table.querySelector("tbody");
    var rows = Array.from(tbody.querySelectorAll("tr.result-row"));

    if (sortState.column === column) {
      sortState.asc = !sortState.asc;
    } else {
      sortState.column = column;
      sortState.asc = true;
    }

    rows.sort(function(a, b) {
      var av = a.getAttribute("data-" + column) || "";
      var bv = b.getAttribute("data-" + column) || "";
      if (column === "duration") {
        av = parseFloat(av) || 0;
        bv = parseFloat(bv) || 0;
        return sortState.asc ? av - bv : bv - av;
      }
      if (av < bv) return sortState.asc ? -1 : 1;
      if (av > bv) return sortState.asc ? 1 : -1;
      return 0;
    });

    // Re-append rows (with their detail rows)
    rows.forEach(function(row) {
      tbody.appendChild(row);
      var detailId = row.getAttribute("data-detail");
      var detail = document.getElementById(detailId);
      if (detail) tbody.appendChild(detail);
    });

    // Update sort arrows
    table.querySelectorAll("th[data-sort]").forEach(function(th) {
      var arrow = th.querySelector(".sort-arrow");
      if (th.getAttribute("data-sort") === column) {
        arrow.textContent = sortState.asc ? " \\u25B2" : " \\u25BC";
      } else {
        arrow.textContent = "";
      }
    });
  }

  document.querySelectorAll("th[data-sort]").forEach(function(th) {
    th.addEventListener("click", function() {
      sortTable(th.getAttribute("data-sort"));
    });
  });

})();
"""


# ---------------------------------------------------------------------------
# HTML section builders
# ---------------------------------------------------------------------------


def _build_header_section(suite: TestSuite) -> str:
    """Build the report header with logo, API info, and donut chart."""
    now = suite.completed_at or datetime.now(timezone.utc)
    ts = now.strftime("%Y-%m-%d %H:%M:%S UTC")

    donut = _donut_svg(suite.pass_rate)

    return (
        '<div class="report-header">'
        '<div class="header-info">'
        f'<div class="logo">agentspec</div>'
        f'<div class="api-name">{_esc(suite.api_name)}</div>'
        f'<div class="api-source">{_esc(suite.api_source)}</div>'
        f'<div class="timestamp">Generated {_esc(ts)}</div>'
        '</div>'
        f'<div class="header-chart">{donut}</div>'
        '</div>'
    )


def _build_summary_cards(suite: TestSuite) -> str:
    """Build the grid of summary statistic cards."""
    critical_count = len(suite.critical_failures)
    critical_class = " critical-alert" if critical_count > 0 else ""

    cards = [
        (
            "card-total", "Total",
            f'<span class="count-up" data-target="{suite.total}">0</span>',
        ),
        (
            "card-pass", "Passed",
            f'<span class="count-up" data-target="{suite.passed}">0</span>',
        ),
        (
            "card-fail", "Failed",
            f'<span class="count-up" data-target="{suite.failed}">0</span>',
        ),
        (
            "card-error", "Errors",
            f'<span class="count-up" data-target="{suite.errors}">0</span>',
        ),
        (
            "card-skip", "Skipped",
            f'<span class="count-up" data-target="{suite.skipped}">0</span>',
        ),
        (
            "card-duration", "Duration",
            f'<span>{_esc(_format_duration(suite.duration_ms))}</span>',
        ),
        (
            f"card-critical{critical_class}", "Critical",
            f'<span class="count-up" data-target="{critical_count}">0</span>',
        ),
    ]

    html_parts = ['<div class="summary-grid">']
    for css_class, label, value_html in cards:
        html_parts.append(
            f'<div class="summary-card {css_class}">'
            f'<div class="card-value">{value_html}</div>'
            f'<div class="card-label">{_esc(label)}</div>'
            f'</div>'
        )
    html_parts.append('</div>')
    return "".join(html_parts)


def _build_category_breakdown(suite: TestSuite) -> str:
    """Build the category breakdown section with bar chart."""
    by_category = suite.by_category()
    if not by_category:
        return ""

    chart_svg = _category_bar_chart_svg(by_category)

    return (
        '<div class="section">'
        '<div class="section-title">Category Breakdown</div>'
        f'<div class="section-body">{chart_svg}</div>'
        '</div>'
    )


def _build_filter_bar(suite: TestSuite) -> str:
    """Build the filter controls above the results table."""
    # Collect present categories, statuses, severities from results.
    categories: set[str] = set()
    statuses: set[str] = set()
    severities: set[str] = set()

    for r in suite.results:
        categories.add(r.category.value)
        statuses.add(r.status.value)
        severities.add(r.severity.value)

    parts: list[str] = ['<div class="filters">']

    # Status filter
    parts.append('<div class="filter-group">')
    parts.append('<span class="filter-label">Status</span>')
    parts.append(
        '<button class="filter-btn active" '
        'data-filter-group="status" data-filter-value="all">All</button>'
    )
    for s in [TestStatus.PASSED, TestStatus.FAILED, TestStatus.ERROR, TestStatus.SKIPPED]:
        if s.value in statuses:
            parts.append(
                f'<button class="filter-btn" '
                f'data-filter-group="status" '
                f'data-filter-value="{_esc(s.value)}">'
                f'{_esc(_STATUS_LABELS[s])}</button>'
            )
    parts.append('</div>')

    # Category filter
    parts.append('<div class="filter-group">')
    parts.append('<span class="filter-label">Category</span>')
    parts.append(
        '<button class="filter-btn active" '
        'data-filter-group="category" data-filter-value="all">All</button>'
    )
    for cat in _CATEGORY_ORDER:
        if cat.value in categories:
            parts.append(
                f'<button class="filter-btn" '
                f'data-filter-group="category" '
                f'data-filter-value="{_esc(cat.value)}">'
                f'{_esc(_CATEGORY_LABELS[cat])}</button>'
            )
    parts.append('</div>')

    # Severity filter
    parts.append('<div class="filter-group">')
    parts.append('<span class="filter-label">Severity</span>')
    parts.append(
        '<button class="filter-btn active" '
        'data-filter-group="severity" data-filter-value="all">All</button>'
    )
    for sev in [
        TestSeverity.CRITICAL, TestSeverity.HIGH, TestSeverity.MEDIUM,
        TestSeverity.LOW, TestSeverity.INFO,
    ]:
        if sev.value in severities:
            parts.append(
                f'<button class="filter-btn" '
                f'data-filter-group="severity" '
                f'data-filter-value="{_esc(sev.value)}">'
                f'{_esc(_SEVERITY_LABELS[sev])}</button>'
            )
    parts.append('</div>')

    parts.append('</div>')
    return "".join(parts)


def _build_result_detail(result: TestResult, detail_id: str) -> str:
    """Build the expandable detail row for a single test result."""
    parts: list[str] = [
        f'<tr class="detail-row" id="{_esc(detail_id)}">',
        '<td class="detail-cell" colspan="6">',
    ]

    # Assertion failures
    if result.failures:
        parts.append('<div class="detail-section">')
        parts.append('<div class="detail-section-title">Assertion Failures</div>')
        for f in result.failures:
            parts.append(
                '<div class="failure-item">'
                f'<span class="failure-assertion">{_esc(f.assertion)}</span>: '
                f'expected <span class="failure-expected">{_esc(f.expected)}</span>'
                f' &mdash; actual <span class="failure-actual">{_esc(f.actual)}</span>'
                '</div>'
            )
        parts.append('</div>')

    # Error message
    if result.error_message:
        parts.append('<div class="detail-section">')
        parts.append('<div class="detail-section-title">Error</div>')
        parts.append(f'<div class="error-msg">{_esc(result.error_message)}</div>')
        parts.append('</div>')

    # Response body preview
    if result.response_body is not None:
        preview = _json_preview(result.response_body)
        if preview:
            parts.append('<div class="detail-section">')
            parts.append('<div class="detail-section-title">Response Body</div>')
            parts.append(f'<pre class="response-preview">{_esc(preview)}</pre>')
            parts.append('</div>')

    parts.append('</td></tr>')
    return "".join(parts)


def _build_results_table(suite: TestSuite) -> str:
    """Build the full results table with sortable columns and expandable rows."""
    if not suite.results:
        return (
            '<div class="section">'
            '<div class="section-title">Test Results</div>'
            '<div class="section-body text-muted">No results to display.</div>'
            '</div>'
        )

    parts: list[str] = [
        '<div class="section">',
        '<div class="section-title">Test Results</div>',
        _build_filter_bar(suite),
        '<div style="overflow-x: auto;">',
        '<table class="results-table" id="resultsTable">',
        '<thead><tr>',
        '<th data-sort="status">Status<span class="sort-arrow"></span></th>',
        '<th>Test Name</th>',
        '<th data-sort="category">Category<span class="sort-arrow"></span></th>',
        '<th data-sort="severity">Severity<span class="sort-arrow"></span></th>',
        '<th>Code</th>',
        '<th data-sort="duration">Duration<span class="sort-arrow"></span></th>',
        '</tr></thead>',
        '<tbody>',
    ]

    for idx, result in enumerate(suite.results):
        detail_id = f"detail-{idx}"
        status_class = _status_css_class(result.status)
        severity_class = _severity_css_class(result.severity)
        icon = _STATUS_ICONS.get(result.status, "?")
        category_label = _CATEGORY_LABELS.get(result.category, result.category.value)
        severity_label = _SEVERITY_LABELS.get(result.severity, result.severity.value)
        status_code_str = str(result.status_code) if result.status_code is not None else "-"
        duration_str = _format_duration(result.duration_ms)

        parts.append(
            f'<tr class="result-row {status_class} {severity_class}" '
            f'data-detail="{_esc(detail_id)}" '
            f'data-status="{_esc(result.status.value)}" '
            f'data-category="{_esc(result.category.value)}" '
            f'data-severity="{_esc(result.severity.value)}" '
            f'data-duration="{result.duration_ms:.2f}">'
            f'<td><span class="status-icon">{icon}</span></td>'
            f'<td>{_esc(result.test_name)}</td>'
            f'<td><span class="badge badge-category">{_esc(category_label)}</span></td>'
            f'<td><span class="badge badge-severity">{_esc(severity_label)}</span></td>'
            f'<td class="text-mono">{_esc(status_code_str)}</td>'
            f'<td class="text-mono">{_esc(duration_str)}</td>'
            f'</tr>'
        )
        parts.append(_build_result_detail(result, detail_id))

    parts.append('</tbody></table></div></div>')
    return "".join(parts)


def _build_fix_suggestions_section(fix_report: Any) -> str:
    """Build the fix suggestions section.

    Parameters
    ----------
    fix_report:
        A ``FixReport`` instance, or ``None``.
    """
    if fix_report is None:
        return ""

    suggestions = fix_report.suggestions
    if not suggestions:
        return ""

    # Group by severity.
    by_severity: dict[str, list[Any]] = {}
    for s in suggestions:
        by_severity.setdefault(s.severity, []).append(s)

    severity_order = ["critical", "high", "medium", "low"]
    severity_labels = {
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }

    parts: list[str] = [
        '<div class="section">',
        '<div class="section-title">'
        f'Fix Suggestions ({fix_report.total_suggestions})'
        '</div>',
        '<div class="section-body">',
    ]

    # Summary
    parts.append(
        f'<p class="fix-description">{_esc(fix_report.summary)}</p>'
    )

    code_counter = 0

    for sev in severity_order:
        group = by_severity.get(sev)
        if not group:
            continue

        label = severity_labels.get(sev, sev.title())
        parts.append(f'<div class="fix-group-title">{_esc(label)} ({len(group)})</div>')

        for suggestion in group:
            code_id = f"fix-code-{code_counter}"
            code_counter += 1

            parts.append('<div class="fix-card">')
            # Header (click to expand)
            parts.append(
                '<div class="fix-card-header">'
                f'<span class="fix-card-title">{_esc(suggestion.title)}</span>'
                f'<span class="fix-card-toggle">&#9660;</span>'
                '</div>'
            )
            # Body
            parts.append('<div class="fix-card-body">')

            # Description
            parts.append(
                f'<div class="fix-description">{_esc(suggestion.description)}</div>'
            )

            # Code suggestion
            if suggestion.code_suggestion:
                parts.append(
                    '<div class="code-block-wrapper">'
                    f'<button class="copy-btn" data-code="{_esc(code_id)}">Copy</button>'
                    f'<pre class="code-block" id="{_esc(code_id)}">'
                    f'{_esc(suggestion.code_suggestion)}</pre>'
                    '</div>'
                )

            # Agent instruction
            if suggestion.agent_instruction:
                parts.append(
                    '<div class="detail-section mt-2">'
                    '<div class="detail-section-title">Agent Instruction</div>'
                    f'<div class="agent-instruction">{_esc(suggestion.agent_instruction)}</div>'
                    '</div>'
                )

            parts.append('</div>')  # fix-card-body
            parts.append('</div>')  # fix-card

    parts.append('</div></div>')
    return "".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class HTMLReporter:
    """Generate self-contained HTML reports from test suite results.

    The report is a single HTML file with all CSS and JavaScript embedded
    inline.  It uses no external resources and can be opened directly in any
    modern browser or attached to a pull request or CI artifact.

    Example
    -------
    ::

        reporter = HTMLReporter()
        html_string = reporter.generate(suite)
        reporter.save(suite, "/tmp/report.html", fix_report=fix_report)
    """

    def __init__(self) -> None:
        pass

    def generate(
        self,
        suite: TestSuite,
        fix_report: Any | None = None,
    ) -> str:
        """Generate a complete self-contained HTML report string.

        Parameters
        ----------
        suite:
            The completed test suite (with results populated).
        fix_report:
            Optional :class:`~agentspec.core.fixer.FixReport` with
            auto-fix suggestions to include in the report.

        Returns
        -------
        str
            A full HTML document as a string.
        """
        header = _build_header_section(suite)
        summary = _build_summary_cards(suite)
        category = _build_category_breakdown(suite)
        results = _build_results_table(suite)
        fixes = _build_fix_suggestions_section(fix_report)

        css = _build_css()
        js = _build_js()

        return (
            '<!DOCTYPE html>\n'
            '<html lang="en" data-theme="light">\n'
            '<head>\n'
            '<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>agentspec Report - {_esc(suite.api_name)}</title>\n'
            f'<style>\n{css}\n</style>\n'
            '</head>\n'
            '<body>\n'
            '<button class="theme-toggle" id="themeToggle">Dark Mode</button>\n'
            '<div class="container">\n'
            f'{header}\n'
            f'{summary}\n'
            f'{category}\n'
            f'{results}\n'
            f'{fixes}\n'
            '</div>\n'
            f'<script>\n{js}\n</script>\n'
            '</body>\n'
            '</html>'
        )

    def save(
        self,
        suite: TestSuite,
        output_path: str,
        fix_report: Any | None = None,
    ) -> str:
        """Generate and save an HTML report to a file.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        suite:
            The completed test suite (with results populated).
        output_path:
            Filesystem path where the HTML file will be written.
        fix_report:
            Optional :class:`~agentspec.core.fixer.FixReport` with
            auto-fix suggestions.

        Returns
        -------
        str
            The absolute path to the written file.
        """
        html_content = self.generate(suite, fix_report=fix_report)

        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_content, encoding="utf-8")

        logger.info("HTML report saved to %s", out)
        return str(out)
