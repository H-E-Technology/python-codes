from dataclasses import dataclass
import math
from collections import defaultdict


@dataclass
class SpeakChunk:
  start_time: int
  end_time: int
  speak_time: int
  speaker: str


@dataclass
class SpeakData:
  window_size_sec: int
  audio_duration: int
  video_frame_length: int
  chunks: list[SpeakChunk] = None
  utterance_amount_map: dict = None
  main_speaker: str = None
  each_speaker_window_utterence: dict = None


  @classmethod
  def from_diarization(cls, diarization, window_size_sec, audio_duration) -> "SpeakData":
    video_frame_length = math.ceil(audio_duration / window_size_sec)
    instance = cls(window_size_sec, audio_duration, video_frame_length)


    chunks = [] # 発話開始時間、発話終了時間、speaker
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        chunks.append(SpeakChunk(start_time=turn.start, end_time=turn.end, speak_time=turn.end - turn.start, speaker=speaker))
    instance.chunks = chunks
    instance.utterance_amount_map = instance._calc_uterence_amount()
    instance.main_speaker = max(instance.utterance_amount_map.items(),key = lambda x:x[1])[0]
    instance.each_speaker_window_utterence = instance._calc_each_speaker_window_utterence()

    print(f"instance: {instance}")
    return instance

  def _calc_uterence_amount(self):
    utterance_amount_map = {}
    for chunk in self.chunks:
      if chunk.speaker not in utterance_amount_map.keys():
        utterance_amount_map[chunk.speaker] = chunk.speak_time
      else:
        utterance_amount_map[chunk.speaker] += chunk.speak_time
    return utterance_amount_map

  def _calc_each_speaker_window_utterence(self):
    # window_time別の各人の発話量
    each_speaker_window_utterence = defaultdict(lambda: defaultdict(float))

    # データを5秒ごとに区切って発話時間を集計
    for chunk in self.chunks:
        start_frame = int(chunk.start_time // self.window_size_sec)  # 何番目のフレームか
        end_frame = int(chunk.end_time // self.window_size_sec)

        for frame in range(start_frame, end_frame + 1):
            # 各フレーム内の発話時間を計算
            segment_start = max(frame * self.window_size_sec, chunk.start_time)
            segment_end = min((frame + 1) * self.window_size_sec, chunk.end_time)
            each_speaker_window_utterence[frame][chunk.speaker] += (segment_end - segment_start)

    # もし誰も話していない時間があれば埋める
    for i in range(self.video_frame_length):
        if i not in each_speaker_window_utterence:
            each_speaker_window_utterence[i] = {self.main_speaker: 0.0} # dummy data
    return each_speaker_window_utterence
