from multiprocessing import Pool

_thread_pool = None

def get_pool():
    global _thread_pool
    
    if _thread_pool is None:
        _thread_pool = Pool(processes=16)
        
    return _thread_pool