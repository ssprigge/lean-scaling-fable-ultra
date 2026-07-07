import textwrap

from leanscale.ablation import strip_haskell_signatures, strip_python_annotations


def test_strip_function_annotations():
    src = textwrap.dedent(
        '''
        def f(x: int, y: "List[str]" = None, *args: int, **kw: str) -> dict:
            z: int = x + 1
            return {"z": z}
        '''
    )
    out = strip_python_annotations(src)
    assert "int" not in out
    assert "dict" not in out
    assert "def f(x, y = None, *args, **kw):" in out
    assert "z = x + 1" in out
    # comments/formatting elsewhere preserved
    assert out.count("\n") == src.count("\n")


def test_strip_preserves_unannotated():
    src = "def g(a, b=2):\n    return a + b\n"
    assert strip_python_annotations(src) == src


def test_strip_multiline_signature():
    src = textwrap.dedent(
        """
        def f(
            x: int,
            y: Mapping[str, int] | None,
        ) -> Iterator[tuple[int, ...]]:
            yield (x,)
        """
    )
    out = strip_python_annotations(src)
    assert "Mapping" not in out and "Iterator" not in out
    assert "x," in out and "y," in out


def test_strip_bare_annassign_kept():
    src = "x: int\ny: int = 3\n"
    out = strip_python_annotations(src)
    # bare annotation (no value) left alone; valued one stripped
    assert "x: int" in out
    assert "y = 3" in out


def test_strip_syntax_error_passthrough():
    src = "def broken(:\n"
    assert strip_python_annotations(src) == src


def test_strip_class_attributes_and_methods():
    src = textwrap.dedent(
        """
        class A:
            n: int = 0

            def m(self, q: float) -> float:
                return q * 2
        """
    )
    out = strip_python_annotations(src)
    assert "n = 0" in out
    assert "def m(self, q):" in out
    compile(out, "<test>", "exec")


def test_stripped_python_still_compiles():
    import ast
    import inspect
    import json as json_mod

    src = inspect.getsource(json_mod)
    out = strip_python_annotations(src)
    ast.parse(out)


def test_haskell_signature_strip():
    src = (
        "module M where\n"
        "\n"
        "add :: Int -> Int -> Int\n"
        "add x y = x + y\n"
        "\n"
        "long ::\n"
        "  Monad m =>\n"
        "  m Int\n"
        "long = pure 3\n"
    )
    out = strip_haskell_signatures(src)
    assert "::" not in out
    assert "add x y = x + y" in out
    assert "long = pure 3" in out
