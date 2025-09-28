from abc import ABC, abstractmethod

import audiocraft.models
import pandas


class NarrationInterface(ABC):

    @abstractmethod
    def generate_script_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Iterrator function to generate the script
        :param df: Input is the dataframe of content having colums ---> Page Number, Cotent, Content Clean
        :return:  Output is the dataframe of content having colums ----> Actor, Dialogue, Emotion, Background Activity

        """

        pass

    @abstractmethod
    def generate_script(self, prev_text: str, current_text: str) -> str:
        """Generate narration audio from text.

        Input Arguments:
            prev_text (str): The previous text segment.
            current_text (str): The current text segment to be narrated.
        Output:
            str: The generated narration script.
            Multiple sentences should be separated by newline.
            Narrator <break> Content <break> Emmotion <break> Backgroup Activites
        """
        pass

    @abstractmethod
    def convert_narration_to_enhanced_narration_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """

        :param df: Input is the dataframe of content having colums ----> Actor, Dialogue, Emotion, Background Activity
        :return: Output will add extra columns having ---> Dialogue Enhanced
        """
        pass

    @abstractmethod
    def convert_narration_to_enhanced_narration(self, current_text: str) -> str:
        """Save the generated narration audio to a file.
        Input Arguments:
            current_text (str): The current text segment to be narrated.
            next_text (str): The next text segment to be narrated.
        Output:
            str: The enhanced narration script.
            Output format is a single line as follows :
            Narrator <break> Content <break> Emmotion <break> Backgroup Activites
        """
        pass

    @abstractmethod
    def convert_background_activites_and_dialogues_to_musical_prompt_iterrator(self,
                                                                               df: pandas.DataFrame) -> pandas.DataFrame:
        """
        Convert background activities and dialogues to a musical prompt iterrator

        :param df:  Input is the dataframe of content having colums ----> Actor, Dialogue, Emotion, Background Activity, Dialogue Enhanced
        :return: Output add new column to it --- Musical Prompt
        """
        pass

    @abstractmethod
    def convert_background_activites_and_dialogues_to_musical_prompt(self, dialogue: str, background_activitiy: str) -> str:
        """Convert background activities and dialogues to a musical prompt.
        Input Arguments:
            dialogue (str): The dialogue text.
            background_activities (str): The background activities description.
        Output:
            str: The musical prompt.
        """
        pass

    @abstractmethod
    def narration_check(self, df: pandas.DataFrame) -> None:
        """Check if the narration is valid."""
        pass

    @abstractmethod
    def convert_text_to_speech_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
            Iterator function to convert the text to speech
        :param df:
        :return: df:
        """
        pass

    @abstractmethod
    def generate_background_music_iterator(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """
            Iterator to generate background music from Musical Prompt
        :param df:
        :return:
        """
        pass

    @abstractmethod
    def generate_background_music(self, prompt: str, duration: float, output_path: str, model : audiocraft.models.MusicGen, device: str ) -> float:
        """Generate music based on a prompt and save it to a file.
        Input Arguments:
            prompt (str): The musical prompt for generation.
            duration (float): Duration of the generated music in seconds.
            output_path (str): Path to save the generated music audio file.
            model (audiocraft.models.MusicGen): The MusicGen model to use for generation.
            device (str): The device to run the model on (e.g., "cpu" or "cuda").
        Output:
            float: Duration of the generated music in seconds.
        """
        pass

    @abstractmethod
    def convert_text_to_speech(self, text: str, reference_wav: str, output_path: str, emotion: str, language: str = "en") -> None:
        """Convert text to speech and save it to a file.
        Input Arguments:
            text (str): The text to be converted to speech.
            reference_wav (str): Path to the reference WAV file for voice style.
            output_path (str): Path to save the generated speech audio file.
            language (str): Language code for the text-to-speech conversion. Default is "en" (English).
        """
        pass

    @abstractmethod
    def get_wav_duration(self, wav_path: str) -> float:
        """Get the duration of a WAV audio file.
        Input Arguments:
            wav_path (str): Path to the WAV audio file.
        Output:
            float: Duration of the audio file in seconds.
        """
        pass





