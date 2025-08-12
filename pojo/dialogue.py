from typing import List


class Dialogue :
    def __init__(self, dialogue_id : str, character_name : str, voice_artist_id, text, language, text_speech_path = None, text_speech_duration = None,
                 emmotions : List[str] = None, backgroud_activities: str = None, background_music_path: str = None):

        self.dialogue_id = dialogue_id
        self.character_name = character_name
        self.voice_artist_id = voice_artist_id
        self.text = text
        self.language = language
        self.text_speech_path = text_speech_path
        self.emmotions = emmotions if emmotions is not None else []
        self.backgroud_activities = backgroud_activities
        self.background_music_path = background_music_path
        self.text_speech_duration = text_speech_duration


    def __str__(self):
        return f"Dialogue(id={self.dialogue_id}, character_name={self.character_name}, voice_artist_id={self.voice_artist_id}, text='{self.text}', language='{self.language}', " \
               f"text_speech_path='{self.text_speech_path}', text_speech_duration={self.text_speech_duration}, " \
               f"emmotions={self.emmotions}, backgroud_activities='{self.backgroud_activities}', " \
               f"background_music_path='{self.background_music_path}')"

    def __repr__(self):
        return self.__str__()