"""Tests for conservative, path-aware change risk assessment."""

from cliv.risk import assess_changes, assess_delete, assess_edit, classify_path


def test_dependency_and_ci_changes_are_high_risk(project_dir):
    dependency = classify_path(project_dir / "pyproject.toml", root=project_dir)
    workflow = classify_path(
        project_dir / ".github" / "workflows" / "ci.yml", root=project_dir
    )

    assert dependency.level == "high"
    assert any("dependency" in reason for reason in dependency.reasons)
    assert workflow.level == "high"
    assert any("CI" in reason for reason in workflow.reasons)


def test_security_code_and_secrets_are_high_or_critical(project_dir):
    security = classify_path(project_dir / "cliv" / "security.py", root=project_dir)
    secret = classify_path(project_dir / ".env", root=project_dir)

    assert security.level == "high"
    assert any("security" in reason for reason in security.reasons)
    assert secret.level == "critical"


def test_source_docs_and_deletion(project_dir):
    source = classify_path(project_dir / "maths.py", root=project_dir)
    docs = classify_path(project_dir / "README.md", root=project_dir)
    deleted = assess_delete(project_dir / "old.py", root=project_dir)

    assert source.level == "medium"
    assert docs.level == "low"
    assert deleted.level == "high"
    assert any("deletes" in reason for reason in deleted.reasons)


def test_removing_tests_or_assertions_raises_risk(project_dir):
    old = (
        "def test_one():\n"
        "    assert True\n\n"
        "def test_two():\n"
        "    assert True\n"
    )
    new = "def test_one():\n    assert True\n"

    assessment = assess_edit(
        project_dir / "tests" / "test_maths.py", old, new, root=project_dir
    )

    assert assessment.level == "high"
    assert any("test function" in reason for reason in assessment.reasons)


def test_large_deletion_is_high_risk(project_dir):
    old = "\n".join(f"line {index}" for index in range(300)) + "\n"
    new = "line 0\n"

    assessment = assess_edit(project_dir / "data.txt", old, new, root=project_dir)

    assert assessment.level == "high"
    assert any("large deletion" in reason for reason in assessment.reasons)


def test_unsafe_code_addition_is_high_risk(project_dir):
    old = "def run():\n    return 1\n"
    new = "import subprocess\n\ndef run():\n    subprocess.run(['echo', 'hi'])\n"

    assessment = assess_edit(project_dir / "runner.py", old, new, root=project_dir)

    assert assessment.level == "high"
    assert any("subprocess" in reason for reason in assessment.reasons)


def test_combined_risk_uses_the_highest_file_and_labels_paths(project_dir):
    changes = [
        (project_dir / "README.md", "old\n", "new\n"),
        (project_dir / "pyproject.toml", "a = 1\n", "a = 2\n"),
    ]

    assessment = assess_changes(changes, root=project_dir)

    assert assessment.level == "high"
    assert any("pyproject.toml" in reason for reason in assessment.reasons)
    assert "RISK: HIGH" in assessment.render()
