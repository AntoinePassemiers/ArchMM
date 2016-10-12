# -*- coding: utf-8 -*-

import time

RELEASE_MODE = False


def timeit(func):
    def timed(*args, **kw):
        t0 = time.time()
        result = func(*args, **kw)
        t1 = time.time()
        print '%r (%r, %r) %5.2f millisec' % (func.__name__, args, kw, round((t1 - t0) * 1000))
        return result
    return timed

