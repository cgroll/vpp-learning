#!/usr/bin/env python3
"""Run once after cloning from the template to rename the package and wire up paths.

    python init_project.py

The project name is read from the git remote URL automatically.
You will be prompted for a package abbreviation and optionally a GitHub username.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def get_project_name() -> str:
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL,
            cwd=ROOT,
        ).strip()
        # works for both https://github.com/user/repo.git and git@github.com:user/repo.git
        name = url.rstrip("/").split("/")[-1].split(":")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        if name:
            return name
    except subprocess.CalledProcessError:
        pass
    return ROOT.name


def ask_abbreviation(project_name: str) -> str:
    print(f"\nProject name : {project_name}")
    print("Choose a short Python package abbreviation used in import statements.")
    print("  financial-market-returns → fmr")
    print("  world-of-energy          → woe")
    print("  climate-analysis         → ca")
    while True:
        abbrev = input("\nPackage abbreviation: ").strip().lower()
        if not abbrev:
            print("Cannot be empty.")
            continue
        if not abbrev.isidentifier():
            print(f"  '{abbrev}' is not a valid Python identifier (letters, digits, underscores only).")
            continue
        if abbrev == "pkg":
            print("  'pkg' is the template placeholder — pick something project-specific.")
            continue
        confirm = input(f"  Use '{abbrev}'? [y/N] ").strip().lower()
        if confirm == "y":
            return abbrev


def ask_github_username() -> str | None:
    print("\nGitHub username (used in book/myst.yml for the GitHub link).")
    print("Press Enter to skip if you only want to build the book locally.")
    username = input("\nGitHub username [skip]: ").strip()
    return username if username else None


def replace_in_file(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if old not in text:
        return False
    path.write_text(text.replace(old, new))
    return True


def title_from_kebab(name: str) -> str:
    return name.replace("-", " ").title()


def main() -> None:
    project_name = get_project_name()
    abbrev = ask_abbreviation(project_name)
    github_username = ask_github_username()
    title = title_from_kebab(project_name)

    github_label = f"{github_username}/{project_name}" if github_username else "local only"
    print(f"\nInitializing '{project_name}' (package: '{abbrev}', github: {github_label}) …\n")

    # 1. Rename pkg/ → <abbrev>/
    shutil.move(str(ROOT / "pkg"), str(ROOT / abbrev))
    print(f"  pkg/  →  {abbrev}/")

    # 2. pyproject.toml
    f = ROOT / "pyproject.toml"
    replace_in_file(f, 'name = "my-project"', f'name = "{project_name}"')
    replace_in_file(f, 'packages = ["pkg"]', f'packages = ["{abbrev}"]')
    print("  pyproject.toml updated")

    # 3. pipeline scripts
    for py_file in (ROOT / "pipeline").glob("*.py"):
        replace_in_file(py_file, "from pkg.", f"from {abbrev}.")
    print("  pipeline/ imports updated")

    # 5. book/myst.yml
    f = ROOT / "book" / "myst.yml"
    text = f.read_text()
    text = text.replace("My Research Project", title)
    if github_username:
        text = re.sub(r"github: .+", f"github: {github_username}/{project_name}", text)
    else:
        # Remove the github line entirely so MyST doesn't show a broken link
        text = re.sub(r"\s*github: .+\n", "\n", text)
    f.write_text(text)
    print("  book/myst.yml updated")

    # 6. README headline
    f = ROOT / "README.md"
    replace_in_file(f, "# Project Book Template", f"# {title}")
    print("  README.md updated")

    # 7. Stage everything and commit
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Initialize project: {project_name} (pkg: {abbrev})"],
        cwd=ROOT,
        check=True,
    )
    print("\n  Changes committed.")

    # 8. Remove this script (it should not exist in the final project)
    Path(__file__).unlink()
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Remove init_project.py"],
        cwd=ROOT,
        check=True,
    )

    print(f"""
Done! Next steps:

  uv sync
  make dry-run
  make run

Import paths in pipeline scripts:
  from {abbrev}.paths import ProjPaths
""")


if __name__ == "__main__":
    main()
