import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        in_kw = 'log_time' in kw
        if in_kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int(0.5 + (te - ts) * 1000)
        if not in_kw or kw.get('log_verbose') is True:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed