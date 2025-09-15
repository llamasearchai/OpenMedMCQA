from typer.testing import CliRunner
from medagent.cli import app


def test_menu_help():
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "MedMCQA Agentic RAG CLI" in res.stdout


