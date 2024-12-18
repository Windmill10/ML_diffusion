from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from pydub import AudioSegment

# Load and initialize the BirdNET-Analyzer models.
analyzer = Analyzer()


AudioSegment.from_wav("sparraw.wav").export("sparraw.mp3", format="mp3")

recording = Recording(
    analyzer,
    "sparraw.mp3"
)
recording.analyze()
print(recording.detections)