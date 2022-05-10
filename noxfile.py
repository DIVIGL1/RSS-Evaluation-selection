from typing import Any
import nox


@nox.session
def flake8(session: Any) -> (None):
    session.run("flake8", "src/rss9module/train.py", external=True)
    session.run("flake8", "src/rss9module/data.py", external=True)
    session.run("flake8", "src/rss9module/pipeline.py", external=True)
    session.run("flake8", "src/rss9module/__init__.py", external=True)
    session.run("flake8", "test/test_all.py", external=True)
    session.run("flake8", "setup.py", external=True)


@nox.session
def black(session: Any) -> (None):
    session.run("black", ".", external=True)


@nox.session
def mypy(session: Any) -> (None):
    session.run("mypy", ".", external=True)
    # session.run("mypy", "src/rss9module/train.py", external=True)
    # session.run("mypy", "src/rss9module/data.py", external=True)
    # session.run("mypy", "src/rss9module/pipeline.py", external=True)


@nox.session
def tests(session: Any) -> (None):
    session.run("pytest", external=True)
