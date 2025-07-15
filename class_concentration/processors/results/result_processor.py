from models.persons.tracked_person import TrackedPerson
from processors.score_processor import ScoreProcessor
from processors.results.output_video_processor import OutputVideoProcessor
from models.settings.settings import Settings


class ResultProcessor:
    def __init__(
        self,
        score_processor: ScoreProcessor,
        output_video_processor: OutputVideoProcessor,
        video_score_list: list,
        audio_score_list: list,
        settings: Settings,
    ):
        self.score_processor = score_processor
        self.output_video_processor = output_video_processor

        self.video_score_list = video_score_list
        self.audio_score_list = audio_score_list
        self.settings = settings

    def draw_output_video(
        self,
        tracked_people_xys: dict[int, TrackedPerson],
        tracked_person_keypoints_list: list,
    ):
        self.output_video_processor.draw(
            tracked_people_xys,
            tracked_person_keypoints_list,
            self.video_score_list,
            self.audio_score_list,
        )

    def write_csv(self):
        self.score_processor.write_csv()
