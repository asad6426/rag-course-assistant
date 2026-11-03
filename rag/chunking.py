"""Split Markdown files into one chunk per section, keeping the source."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.source} > {self.heading}"


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = None
    lines: list[str] = []

    def flush() -> None:
        text = " ".join(lines).strip()
        if heading and text:
            chunks.append(Chunk(source, heading, text))

    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            flush()
            heading, lines = line[3:].strip(), []
        elif line and not line.startswith("# "):
            lines.append(line)
    flush()
    return chunks


def load_directory(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for file in sorted(Path(path).glob("*.md")):
        chunks.extend(chunk_markdown(file.name, file.read_text(encoding="utf-8")))
    return chunks
