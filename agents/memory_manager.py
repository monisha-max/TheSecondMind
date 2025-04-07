import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MemoryManager:
    """
    Manages short-term memory and logs events.
    Consider using Redis or a database for production use.
    """
    def __init__(self):
        self.short_term_memory = {}
        self.logs = []

    def store_data(self, key, value):
        self.short_term_memory[key] = value
        self.log_event(f"Stored key: {key}")

    def get_data(self, key):
        return self.short_term_memory.get(key, None)

    def log_event(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        logging.info(message)

    def get_logs(self):
        return self.logs
