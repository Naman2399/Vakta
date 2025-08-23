from abc import ABC, abstractmethod

import pandas


class OCRInterface(ABC):

    @abstractmethod
    def extract_text_from_pdf(self, image_path: str, start_page: int, end_page: int, output_csv_file_name: str) -> pandas.DataFrame:
        """Extract text from an image file.
        Input Arguments:
            image_path (str): Path to the image file.
            start_page (int): Starting page number for extraction.
            end_page (int): Ending page number for extraction.
            output_csv_file_name (str): Name of the output CSV file containing extracted text.

        Returns:
            pandas.DataFrame: DataFrame containing the extracted text.
            This will contain two columns: 'page_number' and 'content'.
        """
        pass

    @abstractmethod
    def clean_ocr_text_iterrator(self, df: pandas.DataFrame) -> pandas.DataFrame:
        '''
        Iterrator function to clean the text in dataframe
        :param df: Input Dataframe will contain columns "Page Number", "Content"
        :return: Add new column to dataframe "Content Clean"
        '''
        pass

    @abstractmethod
    def clean_ocr_text(self, text: str) -> str:
        """Clean and format the extracted OCR text.
        Input Arguments:
            text (str): The raw text extracted from OCR.
        Returns:
            str: The cleaned and formatted text.
        """
        pass

