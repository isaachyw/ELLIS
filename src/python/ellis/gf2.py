"""GF(2) helpers: measurement records as Python ints, and an incremental span (row-echelon basis)."""
import numpy as np


def pack(record):
    """One bool record -> int, bit i = record[i]."""
    return int.from_bytes(np.packbits(record, bitorder="little").tobytes(), "little")


def unpack(record, count):
    """Inverse of pack for a record of `count` bits."""
    buf = np.frombuffer(record.to_bytes((count + 7) // 8, "little"), dtype=np.uint8)
    return np.unpackbits(buf, bitorder="little", count=count).astype(bool)


def to_ints(rows):
    packed = np.packbits(rows.astype(np.uint8), axis=1)
    return [int.from_bytes(r.tobytes(), "big") for r in packed]


class Span:
    def __init__(self):
        self.piv = {}                      # leading bit -> row
    def reduce(self, v):
        while v:
            h = v.bit_length() - 1
            if h not in self.piv:
                return v
            v ^= self.piv[h]
        return 0
    def add(self, v):
        v = self.reduce(v)
        if v:
            self.piv[v.bit_length() - 1] = v
            return True
        return False
    @property
    def dim(self):
        return len(self.piv)
