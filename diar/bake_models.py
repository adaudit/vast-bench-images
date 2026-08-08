import os

from nemo.collections.asr.models import SortformerEncLabelModel
from pyannote.audio import Pipeline

token = os.environ["HF_TOKEN"]
SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2.1")
if Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token) is None:
    raise RuntimeError("pyannote model could not be downloaded; accept its Hugging Face license")
