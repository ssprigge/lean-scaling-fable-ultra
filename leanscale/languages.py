"""Per-language corpus configuration.

Each language names its file extensions, the line-comment prefix used to
render file-boundary headers inside measurement windows, and directory /
filename patterns excluded from the corpus (vendored code, generated code,
compiler test suites full of deliberately-broken code, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LanguageConfig:
    name: str
    extensions: tuple[str, ...]
    comment_prefix: str  # line-comment prefix used for window headers
    block_comment: tuple[str, str] | None = None
    # Extra per-language file basename patterns to exclude (lowercased substring match).
    exclude_file_substrings: tuple[str, ...] = ()


# Directories excluded for every language. Test suites are excluded uniformly:
# compiler/library test suites (lean4/tests, sqlite/test, ...) are full of
# deliberately pathological or broken code that would corrupt a
# "predictability of working code" measurement, so we restrict the corpus to
# library/application code for all languages alike.
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        "test",
        "tests",
        "testing",
        "__tests__",
        "spec",
        "specs",
        "benchmark",
        "benchmarks",
        "vendor",
        "vendored",
        "third_party",
        "3rdparty",
        "thirdparty",
        "extern",
        "external",
        "node_modules",
        "dist",
        "build",
        "out",
        "generated",
        "gen",
        "fixtures",
        "testdata",
        "examples",
        "example",
        "docs",
        "doc",
        "contrib",
        "deps",
        "tools",
        "scripts",
        "ci",
        ".github",
    }
)

MIN_FILE_BYTES = 200
MAX_FILE_BYTES = 262_144  # skip monoliths / bundles / generated giants


LANGUAGES: dict[str, LanguageConfig] = {
    "lean": LanguageConfig(
        name="lean",
        extensions=(".lean",),
        comment_prefix="--",
        block_comment=("/-", "-/"),
    ),
    "python": LanguageConfig(
        name="python",
        extensions=(".py",),
        comment_prefix="#",
        exclude_file_substrings=("_pb2", "parsetab"),
    ),
    "haskell": LanguageConfig(
        name="haskell",
        extensions=(".hs",),
        comment_prefix="--",
        block_comment=("{-", "-}"),
    ),
    "rust": LanguageConfig(
        name="rust",
        extensions=(".rs",),
        comment_prefix="//",
        block_comment=("/*", "*/"),
    ),
    "c": LanguageConfig(
        name="c",
        extensions=(".c", ".h"),
        comment_prefix="//",
        block_comment=("/*", "*/"),
        exclude_file_substrings=(".gen.", "_gen.", "ragel"),
    ),
    "javascript": LanguageConfig(
        name="javascript",
        extensions=(".js", ".mjs", ".cjs"),
        comment_prefix="//",
        block_comment=("/*", "*/"),
        exclude_file_substrings=(".min.", "bundle", "dist"),
    ),
}


def header_line(cfg: LanguageConfig, display_path: str) -> str:
    """File-boundary header rendered as a native comment of the language."""
    if cfg.block_comment and cfg.name in ("c",):
        o, c = cfg.block_comment
        return f"{o} ==== File: {display_path} ==== {c}\n"
    return f"{cfg.comment_prefix} ==== File: {display_path} ====\n"
