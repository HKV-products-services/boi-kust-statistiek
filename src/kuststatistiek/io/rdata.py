"""
Minimal reader for .RData files (R serialization format 2/3, XDR) for objects that pyreadr cannot
convert, e.g. numeric matrices with dimnames such as Table22h/Table30 in hoekvanh_winddirs.RData.
Supports NULL, symbols, pairlists, character/logical/integer/double vectors and lists; numeric
objects are returned as numpy arrays shaped by their "dim" attribute (column-major, as in R).
"""

import bz2
import gzip
import lzma
import struct

import numpy as np

_NA_INT = -2147483648


class _Reader:
    def __init__(self, buf):
        self.buf, self.pos, self.refs = buf, 0, []

    def int(self):
        v = struct.unpack_from(">i", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def bytes(self, n):
        b = self.buf[self.pos : self.pos + n]
        self.pos += n
        return b

    def length(self):
        n = self.int()
        if n == -1:  # long vector
            n = (self.int() << 32) + self.int()
        return n

    def item(self, flags=None):
        flags = self.int() if flags is None else flags
        t, has_attr, has_tag = (
            flags & 0xFF,
            bool(flags & (1 << 9)),
            bool(flags & (1 << 10)),
        )
        if t in (254, 242):  # NILVALUE_SXP, EMPTYENV
            return None
        if t == 255:  # REFSXP
            return self.refs[(flags >> 8) - 1]
        if t == 1:  # SYMSXP
            name = self.item()
            self.refs.append(name)
            return name
        if t == 238:  # ALTREP_SXP (compact sequences, deferred strings, wrappers)
            info, state, attrs = self.item(), self.item(), self.item()
            cls = list(info.values())[0] if isinstance(info, dict) else info
            if cls in ("compact_intseq", "compact_realseq"):
                n, start, step = (float(x) for x in state[:3])
                v = start + step * np.arange(int(n))
            elif cls == "deferred_string":
                arg = list(state.values())[0] if isinstance(state, dict) else state
                v = [
                    repr(int(x)) if float(x).is_integer() else repr(float(x))
                    for x in np.ravel(arg)
                ]
            elif cls.startswith("wrap_"):
                v = state[0]
            else:
                raise NotImplementedError(f"ALTREP class {cls} not supported")
            if attrs and isinstance(v, np.ndarray) and "dim" in attrs:
                v = v.reshape([int(d) for d in attrs["dim"]], order="F")
            return v
        if t == 9:  # CHARSXP
            n = self.int()
            return None if n == -1 else self.bytes(n).decode("utf-8", "replace")
        if t in (2, 6):  # LISTSXP / LANGSXP (pairlist)
            out = {}
            while t in (2, 6):
                if has_attr:
                    self.item()
                tag = self.item() if has_tag else f"[{len(out) + 1}]"
                out[tag] = self.item()
                flags = (
                    self.int()
                )  # CDR: next cell, NILVALUE_SXP or (dotted pair) any object
                t, has_attr, has_tag = (
                    flags & 0xFF,
                    bool(flags & (1 << 9)),
                    bool(flags & (1 << 10)),
                )
            if t != 254:
                out[f"[{len(out) + 1}]"] = self.item(flags)
            return out
        if t in (10, 13):  # LGLSXP, INTSXP
            n = self.length()
            v = np.frombuffer(self.bytes(4 * n), dtype=">i4").astype(float)
            v[v == _NA_INT] = np.nan
        elif t == 14:  # REALSXP
            n = self.length()
            v = np.frombuffer(self.bytes(8 * n), dtype=">f8").astype(float)
        elif t == 16:  # STRSXP
            v = [self.item() for _ in range(self.length())]
        elif t in (19, 20):  # VECSXP, EXPRSXP
            v = [self.item() for _ in range(self.length())]
        else:
            raise NotImplementedError(f"R object type {t} not supported")
        attrs = self.item() if has_attr else None
        if attrs:
            if isinstance(v, np.ndarray) and "dim" in attrs:
                v = v.reshape([int(d) for d in attrs["dim"]], order="F")
            elif isinstance(v, list) and "names" in attrs:
                v = dict(zip(attrs["names"], v))
        return v


def read_rdata(path):
    """load(path) -> dict {object name: value}."""
    raw = open(path, "rb").read()
    for dec in (gzip.decompress, bz2.decompress, lzma.decompress, lambda b: b):
        try:
            buf = dec(raw)
            break
        except Exception:
            continue
    if not buf.startswith(b"RDX2\nX\n") and not buf.startswith(b"RDX3\nX\n"):
        raise ValueError(f"{path}: not an XDR RData file")
    r = _Reader(buf[7:])
    version = r.int()
    r.int()
    r.int()  # writer / minimal reader version
    if version == 3:
        r.bytes(r.int())  # native encoding
    return r.item()
