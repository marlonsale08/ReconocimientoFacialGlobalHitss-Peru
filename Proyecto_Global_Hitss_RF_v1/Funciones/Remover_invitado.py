import os
import os.path
import time

def removeGuess(origin):
	#old = 5 * 60
    now = time.time()
    files = os.listdir(origin)
    for name in files:
        full_path = os.path.join(origin, name)
        if os.path.isfile(full_path):
            stat = os.stat(full_path)
            if now - stat.st_ctime > old:
                os.remove(full_path)