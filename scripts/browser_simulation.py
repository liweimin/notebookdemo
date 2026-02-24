from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app


ARTIFACTS_DIR = (ROOT_DIR / "artifacts/browser-sim").resolve()
SCREENSHOTS_DIR = ARTIFACTS_DIR / "screenshots"
SAMPLE_DOCS_DIR = ARTIFACTS_DIR / "sample-docs"
REPORT_HTML = ARTIFACTS_DIR / "index.html"
REPORT_JSON = ARTIFACTS_DIR / "run.json"

HOST = "127.0.0.1"
PORT = 8010
BASE_URL = f"http://{HOST}:{PORT}"


@dataclass
class StepRecord:
    name: str
    note: str
    screenshot: str
    timestamp: str


def _prepare_dirs() -> None:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _create_sample_document() -> Path:
    sample = SAMPLE_DOCS_DIR / "core-flow-notes.txt"
    sample.write_text(
        (
            "NotebookLM style system capabilities:\n"
            "1. Upload source materials into a notebook.\n"
            "2. Retrieve relevant chunks by semantic similarity.\n"
            "3. Generate answers grounded by citations from the source material.\n"
            "4. Return concise summaries for quick understanding.\n"
        ),
        encoding="utf-8",
    )
    return sample


def _start_server() -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app=app, host=HOST, port=PORT, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            return server, thread
        time.sleep(0.1)
    raise RuntimeError("Uvicorn server did not start in time.")


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


def _capture(page: Page, steps: list[StepRecord], name: str, note: str, index: int) -> None:
    screenshot_name = f"{index:02d}-{name}.png"
    screenshot_path = SCREENSHOTS_DIR / screenshot_name
    page.screenshot(path=str(screenshot_path), full_page=True)
    steps.append(
        StepRecord(
            name=name,
            note=note,
            screenshot=f"screenshots/{screenshot_name}",
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
    )


def _assert_text_not_contains(text: str, keyword: str, label: str) -> None:
    if keyword in text:
        raise AssertionError(f"{label} still contains '{keyword}'.")


def _run_browser_flow(
    playwright: Playwright,
    sample_doc: Path,
    headed: bool,
    slow_mo: int,
    step_delay_ms: int,
    hold_seconds: int,
) -> dict[str, Any]:
    browser = playwright.chromium.launch(headless=not headed, slow_mo=slow_mo)
    context: BrowserContext = browser.new_context(
        viewport={"width": 1520, "height": 920},
        record_video_dir=str(ARTIFACTS_DIR),
        record_video_size={"width": 1280, "height": 720},
    )
    page = context.new_page()
    steps: list[StepRecord] = []
    notebook_name = f"core-flow-{int(time.time())}"
    video_path = ""
    engine_tag_text = ""

    report: dict[str, Any] = {}
    try:
        page.goto(BASE_URL, wait_until="networkidle")
        if step_delay_ms > 0:
            page.wait_for_timeout(step_delay_ms)
        _capture(page, steps, "open-home", "Open the app home page.", 1)

        page.fill("#notebookNameInput", notebook_name)
        page.click("#createNotebookForm button[type='submit']")
        page.wait_for_selector(f"#notebookList button:has-text('{notebook_name}')")
        if step_delay_ms > 0:
            page.wait_for_timeout(step_delay_ms)
        _capture(page, steps, "create-notebook", f"Created notebook: {notebook_name}.", 2)

        page.set_input_files("#fileInput", str(sample_doc))
        page.click("#uploadForm button[type='submit']")
        page.wait_for_function(
            "document.querySelector('#answerOutput')?.textContent?.includes('文档处理完成，可以提问了。')",
            timeout=180000,
        )
        page.wait_for_selector("#documentList li strong:has-text('core-flow-notes.txt')")
        if step_delay_ms > 0:
            page.wait_for_timeout(step_delay_ms)
        _capture(page, steps, "upload-document", "Uploaded and indexed one sample document.", 3)

        page.fill("#questionInput", "这个系统的核心能力是什么？")
        page.click("#askForm button[type='submit']")
        page.wait_for_function(
            "document.querySelectorAll('#citationList li').length > 0",
            timeout=180000,
        )
        if step_delay_ms > 0:
            page.wait_for_timeout(step_delay_ms)
        answer_text = page.inner_text("#answerOutput")
        _assert_text_not_contains(answer_text, "还没有回答。", "Answer area")
        _assert_text_not_contains(answer_text, "请求失败", "Answer area")
        citation_count = page.locator("#citationList li").count()
        if citation_count < 1:
            raise AssertionError("Expected at least one citation in #citationList.")
        engine_tag_text = page.inner_text("#engineModeTag").strip()
        _capture(page, steps, "ask-question", "Asked a question and received grounded answer with citations.", 4)

        _capture(page, steps, "final-state", "Final state after full browser simulation.", 5)
        if headed and hold_seconds > 0:
            page.wait_for_timeout(hold_seconds * 1000)
        report = {
            "ok": True,
            "base_url": BASE_URL,
            "notebook_name": notebook_name,
            "answer_preview": answer_text[:240],
            "citation_count": citation_count,
            "steps": [s.__dict__ for s in steps],
            "video": video_path,
            "engine_mode_tag": engine_tag_text,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
        return report
    finally:
        try:
            video_obj = page.video
            context.close()
            if video_obj is not None:
                video_path = Path(video_obj.path()).name
                if report:
                    report["video"] = video_path
        finally:
            browser.close()


def _render_html_report(report: dict[str, Any]) -> str:
    status_text = "PASS" if report.get("ok") else "FAIL"
    status_class = "pass" if report.get("ok") else "fail"
    steps_html = "\n".join(
        [
            (
                "<article class='step'>"
                f"<h3>Step {idx + 1}: {step['name']}</h3>"
                f"<p>{step['note']}</p>"
                f"<p class='time'>{step['timestamp']}</p>"
                f"<img src='{step['screenshot']}' alt='{step['name']}' />"
                "</article>"
            )
            for idx, step in enumerate(report.get("steps", []))
        ]
    )
    video_html = ""
    if report.get("video"):
        video_html = (
            "<section class='video'>"
            "<h2>Playback Video</h2>"
            f"<video controls src='{report['video']}'></video>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Browser Simulation Report</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background: #f3f7ee;
      color: #1b2a18;
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .summary {{
      background: #fff;
      border: 1px solid #cfdcc5;
      border-radius: 14px;
      padding: 16px 18px;
      margin-bottom: 20px;
    }}
    .status {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 700;
    }}
    .status.pass {{
      color: #205d20;
      background: #d9efd2;
      border: 1px solid #b8ddb1;
    }}
    .status.fail {{
      color: #7a1f1f;
      background: #f6dada;
      border: 1px solid #e8b8b8;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    .step {{
      background: #fff;
      border: 1px solid #cfdcc5;
      border-radius: 12px;
      padding: 12px;
    }}
    .step h3 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    .step p {{
      margin: 0 0 8px;
      line-height: 1.45;
    }}
    .step .time {{
      color: #5d7059;
      font-size: 13px;
    }}
    .step img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid #d7e3cd;
    }}
    .video {{
      margin-top: 24px;
      background: #fff;
      border: 1px solid #cfdcc5;
      border-radius: 12px;
      padding: 12px;
    }}
    .video video {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid #d7e3cd;
    }}
    code {{
      background: #eef4e8;
      padding: 1px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="summary">
      <h1>Mini NotebookLM Browser Simulation</h1>
      <p>
        Status:
        <span class="status {status_class}">{status_text}</span>
      </p>
      <p>Base URL: <code>{report.get("base_url", "")}</code></p>
      <p>Notebook: <code>{report.get("notebook_name", "")}</code></p>
      <p>Citation count: <code>{report.get("citation_count", 0)}</code></p>
      <p>Answer preview: {report.get("answer_preview", "")}</p>
      <p>Finished at: <code>{report.get("finished_at", "")}</code></p>
    </section>
    <section class="grid">
      {steps_html}
    </section>
    {video_html}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser simulation for Mini NotebookLM.")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser in visible window mode.",
    )
    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        help="Playwright slow_mo in milliseconds.",
    )
    parser.add_argument(
        "--step-delay-ms",
        type=int,
        default=0,
        help="Extra wait after each major step to make actions easier to watch.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="How long to keep final page visible before closing browser.",
    )
    args = parser.parse_args()

    _prepare_dirs()
    sample_doc = _create_sample_document()
    server, thread = _start_server()
    report: dict[str, Any]

    try:
        with sync_playwright() as playwright:
            report = _run_browser_flow(
                playwright=playwright,
                sample_doc=sample_doc,
                headed=args.headed,
                slow_mo=args.slow_mo,
                step_delay_ms=args.step_delay_ms,
                hold_seconds=args.hold_seconds,
            )
    except Exception as exc:
        report = {
            "ok": False,
            "base_url": BASE_URL,
            "error": str(exc),
            "steps": [],
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }
    finally:
        _stop_server(server, thread)

    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    REPORT_HTML.write_text(_render_html_report(report), encoding="utf-8")
    print(f"Simulation report written to: {REPORT_HTML}")
    print(f"Raw report json written to: {REPORT_JSON}")
    if not report.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
