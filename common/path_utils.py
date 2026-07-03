from pathlib import Path
import os
import sys
from typing import Optional, Union


def default_var_root() -> Path:
\
\
\
\
\
       
    env_root = os.environ.get("VAR_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return Path(__file__).resolve().parents[1]


def add_var_root(var_root: Optional[Union[str, os.PathLike]] = None) -> Path:
                                                                               
    root = Path(var_root).expanduser().resolve() if var_root else default_var_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
