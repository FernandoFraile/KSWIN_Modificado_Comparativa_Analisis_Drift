from multiprocessing import Manager


manager = Manager()

excel_lock = manager.Lock()
