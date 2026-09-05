"""Small blocking pool for independent Parakeet model lanes."""
import queue


class ParakeetLane:
    def __init__(self, model, stream):
        self.model = model
        self.stream = stream


class ParakeetPool:
    def __init__(self, instances, loader):
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        cuda = torch is not None and torch.cuda.is_available()
        self.instances = [ParakeetLane(loader(), torch.cuda.Stream() if cuda else None) for _ in range(instances)]
        self._available = queue.Queue(maxsize=instances)
        for instance in self.instances:
            self._available.put(instance)

    def checkout(self):
        return self._available.get()

    def checkin(self, instance):
        self._available.put(instance)
