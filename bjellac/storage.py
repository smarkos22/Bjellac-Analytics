"""Small file and SQL utilities shared by the public source selection."""
import os
from pathlib import Path
import re
import tempfile


def sql_identifier(value: str, *, qualified: bool = False) -> str:
    parts = value.split('.') if qualified else [value]
    if not parts or any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', p) for p in parts):
        raise ValueError('Invalid SQL identifier')
    return '.'.join(parts)


def atomic_frame(frame, path, *, csv=False):
    """Readers observe a complete old or new file, never a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as output:
            (frame.write_csv if csv else frame.write_parquet)(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
