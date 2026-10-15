"""Small blocking pool for independent Parakeet model lanes."""
import queue


class ParakeetPool:
    def __init__(self, instances, loader):
        self.instances = [loader() for _ in range(instances)]
        self._available = queue.Queue(maxsize=instances)
        for instance in self.instances:
            self._available.put(instance)

    def checkout(self):
        return self._available.get()

    def checkin(self, instance):
        self._available.put(instance)
