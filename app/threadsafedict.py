import threading

class ThreadSafeDict:
    def __init__(self):
        self._lock = threading.Lock()
        self._dict = {}

    def set(self, key, value):
        with self._lock:  # Acquire the lock before modifying the dictionary
            self._dict[key] = value

    def get(self, key, default):
        with self._lock:  # Acquire the lock before accessing the dictionary
            return self._dict.get(key, default)

    def remove(self, key):
        with self._lock:  # Acquire the lock before modifying the dictionary
            if key in self._dict:
                del self._dict[key]

    def items(self):
        with self._lock:  # Acquire the lock to safely iterate over items
            return list(self._dict.items())

    def __len__(self):
        with self._lock:  # Acquire the lock to safely get the length
            return len(self._dict)