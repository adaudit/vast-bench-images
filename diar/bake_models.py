import os

from nemo.collections.asr.models import SortformerEncLabelModel
import torch
from pyannote.audio import Pipeline

token = os.environ["HF_TOKEN"]
SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
# Pyannote 3.1.1's version parser rejects NVIDIA's 2.8.0a0 build suffix.
torch.__version__ = torch.__version__.split("a", 1)[0].split("+", 1)[0]
if Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token) is None:
    raise RuntimeError("pyannote model could not be downloaded; accept its Hugging Face license")
