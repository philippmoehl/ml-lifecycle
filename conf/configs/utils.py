import re

import hydra
from omegaconf import OmegaConf

import torch



def get_constant(path: str) -> float:
    try:
        obj = _locate(path)
        if not isinstance(obj, float):
            raise ValueError(
                f"Located non-float of type '{type(obj).__name__}'"
                + f" while loading '{path}'"
            )
        cl: float = obj
        return cl
    except Exception as e:
        raise e


def get_dtype(path: str) -> torch.dtype:
    try:
        obj = _locate(path)
        if not isinstance(obj, torch.dtype):
            raise ValueError(
                f"Located non-torch-type of type '{type(obj).__name__}'"
                + f" while loading '{path}'"
            )
        cl: torch.dtype = obj
        return cl
    except Exception as e:
        raise e


def get_lambda(expr: str):
    """Reads in a string and returns a lambda function.

        Supports torch functions and mathematical expressions.

            Typical example:

            u0: '-2.4 * torch.sin(2x) * torch.cos(2*x)**3'
        """
    expr = re.sub(r"(\d+)x", r"\1*x", expr)  # "<n>x" -> "<n>*x"

    p = re.compile(r"([a-zA-Z0-9]+)(\.[a-zA-Z0-9]+)+")  # finds imports in expr
    for match in re.finditer(p, expr):
        if match.group(1) != "torch":
            raise NameError(
                "At the moment, the u0 function only supports torch methods.")
        match_mod = match.group(0).split(".")[0]
        match_attrs = match.group(0).split(".")[1:]
        _unfold_module(match_mod, match_attrs)
    try:
        f = eval("lambda x:" + expr)
        return f
    except Exception as e:
        raise e


def _locate(path: str):
    """
        Locate an object by name or dotted path, importing as necessary.
        This is similar to the pydoc function `locate`, except that it checks
        for the module from the given path from back to front.
        """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj


def _unfold_module(mod, attrs):
    if hasattr(eval(mod), attrs[0]):
        if len(attrs) > 1:
            return _unfold_module(f"{mod}.{attrs[0]}", attrs[1:])
        else:
            return
    else:
        raise AttributeError(f"Module {mod} has no attribute {'.'.join(attrs)}")


def register_resolvers():
    OmegaConf.register_new_resolver(
        name="get_method",
        resolver=lambda mtd: hydra.utils.get_method(mtd))
    OmegaConf.register_new_resolver(
        name="get_cls",
        resolver=lambda cls: hydra.utils.get_class(cls))
    OmegaConf.register_new_resolver(
        name="get_constant",
        resolver=lambda c: get_constant(c))
    OmegaConf.register_new_resolver(
        name="get_dtype",
        resolver=lambda dtype: get_dtype(dtype))
    OmegaConf.register_new_resolver(
        name="get_lambda",
        resolver=lambda expr: get_lambda(expr))
    OmegaConf.register_new_resolver(
        name="str_",
        resolver=lambda n: f"${{{n}}}"
    )
    OmegaConf.register_new_resolver(
        name="mult",
        resolver=lambda n, m: n * m
    )


def setup_hydra():
    """
    Setup Hydra.
    """
    register_resolvers()

    from . import config
    config.store()
