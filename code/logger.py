
# Project Logger

from miscellaneous import getTime
from datetime import datetime

activeTask = None
def setActive(task):
    newLine()
    global activeTask
    activeTask = task.upper()
    log("Start of the execution of " + activeTask + "!")
def newLine():
    if activeTask: log("")
    else: print("")

def log(word):
    now = datetime.now()
    # [" + str(now.date()) + "]
    print("[" + activeTask + "][" + str(now.strftime("%H:%M:%S")) + "] " + word)