from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_suggestion_populates_composer_without_auto_sending():
    script = (ROOT / "static" / "script.js").read_text(encoding="utf-8")
    function = script.split("function suggest(t){", 1)[1].split("}", 1)[0]
    assert "input.value=t" in function
    assert "input.focus()" in function
    assert "send()" not in function


def test_chat_deletion_and_structured_markdown_ui_are_present():
    script = (ROOT / "static" / "script.js").read_text(encoding="utf-8")
    page = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert "method:'DELETE'" in script
    assert "+ New chat" in page
    assert "<table>" in script and "<h${n}>" in script
    assert "script.js?v=6" in page


def test_measured_upload_and_full_artifact_names_are_visible():
    script = (ROOT / "static" / "script.js").read_text(encoding="utf-8")
    page = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assert 'accept=".idf,.epw,.csv"' in page
    assert "active-measured" in page and "loadArtifacts" in script
    assert 'title="Download ${esc(f.name)}"' in script
    assert 'class="artifact-name"' in script and "fileExtension(f.name)" in script
