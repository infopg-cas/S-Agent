from queue import Queue


class MessagePool:
    def __init__(self):
        self.message_queue = Queue()
        self.history_queue = Queue()
