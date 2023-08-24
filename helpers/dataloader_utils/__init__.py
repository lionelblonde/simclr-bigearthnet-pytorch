from pathlib import Path, PurePath
from typing import List


def path2str(path: Path) -> str:
    return str(path.resolve())


def read_from_file(path: str, parent: str = "") -> List[str]:
    content = []
    with Path(path).open("r") as f:
        content = f.readlines()
    content = [
        str(PurePath(parent).joinpath(line.strip()))
        for line in content if len(line.strip()) > 0
    ]
    return content


def save2file(filepath: Path, content: List[str]) -> None:
    with filepath.resolve().open("w") as f:
        for line in content:
            f.write(f"{line}\n")
