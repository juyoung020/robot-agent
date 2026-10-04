import struct, numpy as np


def load(path, keep_depth=True):
    f = open(path, 'rb')
    assert f.read(4) == b'SGRC'
    struct.unpack('<I', f.read(4))
    G, I, P = [], [], []
    while True:
        try:
            t = f.read(1)
            if not t:
                break
            st, = struct.unpack('<d', f.read(8))
            if t == b'P':
                n, = struct.unpack('<i', f.read(4))
                P.append((st, np.frombuffer(f.read(4 * n), np.float32)))
            elif t == b'G':
                G.append((st,) + struct.unpack('<3d', f.read(24)))
            elif t == b'I':
                w, h = struct.unpack('<ii', f.read(8))
                K = struct.unpack('<4d', f.read(32))
                dep = np.frombuffer(f.read(4 * w * h), np.float32).reshape(h, w)
                has = f.read(1)[0]
                if has == 1:
                    f.seek(w * h * 3, 1)
                n, iw, ih, mw, mh = struct.unpack('<5i', f.read(20))
                ms = struct.unpack('<4f', f.read(16))
                d = dict(st=st, w=w, h=h, K=K, depth=dep if keep_depth else None, n=n, iw=iw, ih=ih, mw=mw, mh=mh, ms=ms)
                if n:
                    words = (mw * mh + 31) // 32
                    d['cls'] = np.frombuffer(f.read(4 * n), np.int32)
                    d['score'] = np.frombuffer(f.read(4 * n), np.float32)
                    d['box'] = np.frombuffer(f.read(16 * n), np.float32).reshape(n, 4)
                    d['bits'] = np.frombuffer(f.read(4 * words * n), np.uint32).reshape(n, words)
                I.append(d)
            else:
                break
        except (struct.error, ValueError, IndexError):
            break
    return P, G, I
