from models.settings.settings import Settings
from models.scores.speak_data import SpeakData
import os
import ffmpeg
from pyannote.audio import Pipeline
from google.colab import userdata
from utils.utils import get_audio_duration
from processors.score_collector import AudioScoreCollector
from config import *


class AudioSimilarityProcessor:
    """話者分離 -> スコア計算は 1.0 - mainspeaker(先生など)以外の発話時間の割合　で計算されている"""

    def __init__(
        self,
        audio_score_collector: AudioScoreCollector,
        settings: Settings,
    ):
        self.video_path = settings.video_info.video_path
        self.window_size_sec = settings.window_size_sec
        self.diarization = None
        self.speak_data = None
        self.audio_duration = None
        self.base_path = settings.base_path

        # hugging face の token が必要
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=userdata.get("HF_TOKEN")
        )

        # ffmpeg を使って音声を抽出
        self.audio_path = os.path.join(self.base_path, "output.wav")
        if os.path.exists(self.audio_path):
            os.remove(self.audio_path)
        ffmpeg.input(self.video_path).output(self.audio_path, format="wav").run()

        self.audio_score_collector = audio_score_collector

    def _dialization(self):
        self.diarization = self.pipeline(self.audio_path)

    def _calc_speak_utterence_main_or_others_per_window(self):
        for window, speaker_utterence_map in sorted(
            self.speak_data.each_speaker_window_utterence.items()
        ):
            # 新しい辞書を作成
            others = [
                k for k in speaker_utterence_map if k != self.speak_data.main_speaker
            ]
            renamed = {
                "main_speaker": speaker_utterence_map[self.speak_data.main_speaker]
            }
            for i, other_key in enumerate(others, start=1):
                renamed[f"other{i}"] = speaker_utterence_map[other_key]
            self.audio_score_collector.speak_utterence_main_or_others_per_window.append(
                renamed
            )

    def run(self) -> list:
        self._dialization()
        audio_duration = get_audio_duration(self.audio_path)

        self.speak_data = SpeakData.from_diarization(
            self.diarization, self.window_size_sec, audio_duration
        )
        self._calc_speak_utterence_main_or_others_per_window()

        for (
            speaker_info
        ) in self.audio_score_collector.speak_utterence_main_or_others_per_window:
            other_voice_amount = 0

            for speaker, amount in speaker_info.items():
                if speaker != "main_speaker":
                    other_voice_amount += amount

            score = max([0.0, 1.0 - other_voice_amount / self.window_size_sec])
            self.audio_score_collector.window_total_audioscore_list.append(score)
