'''
This script extracts text from a PDF file using EasyOCR and saves the extracted text to a CSV file.
It processes a specified range of pages and handles multiple chapters defined in a mapping.
Input : pdf_path, chapter_mapping which contains the start and end pages for each chapter.
Output: CSV files containing the extracted text for each chapter.
'''

import argparse
import csv
import os

import easyocr
import fitz
import numpy as np
from tqdm import tqdm


def get_argument() :

    """Parse command line arguments."""
    ##############################################################################
    # Create an argument parser
    ##############################################################################

    parser = argparse.ArgumentParser(description='Extract text from PDF and save to CSV.')
    parser.add_argument('--pdf_path', type=str, default='books/sample/book.pdf', help='Path to the input PDF file')
    return parser.parse_args()

def extract_text_from_pdf(pdf_path, start_page, end_page, output_csv):
    reader = easyocr.Reader(['en'])
    doc = fitz.open(pdf_path)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['page_number', 'content'])
        for page_num in tqdm(range(start_page - 1, end_page), desc="Processing pages", total = end_page - start_page + 1):
            pix = doc[page_num].get_pixmap(dpi=300)  # Higher DPI for better OCR
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            text = ' '.join(reader.readtext(img_np, detail=0))
            writer.writerow([page_num + 2 - start_page, text.strip()])


if __name__ == "__main__":

    # Parse command line arguments
    args = get_argument()

    book_path = args.pdf_path

    # Define the chapter mapping
    chapter_mapping = { # key is the chapter name, value is a list of start and end pages
        'chapter_1': [5, 9],
    }

    # Ensure the PDF file exists
    if not os.path.isfile(book_path):
        raise FileNotFoundError(f"The specified PDF file does not exist: {book_path}")
    # Ensure start_page and end_page are valid
    for chapter_name, (start_page, end_page) in chapter_mapping.items():
        if not isinstance(start_page, int) or not isinstance(end_page, int):
            raise ValueError(f"Invalid page numbers for {chapter_name}: start_page and end_page must be integers.")
        # Ensure start_page and end_page are positive integers
        if start_page < 1 or end_page < 1 or end_page < start_page:
            raise ValueError(f"Invalid page numbers for {chapter_name}: start_page and end_page must be >= 1.")


    #################################
    # Print the paths and parameters
    #################################
    print("Parameters:")
    print(f"Book path: {book_path}")
    print(f"Input PDF file: {book_path}")
    print(f"Page range: {start_page} to {end_page}")
    print("-"*100)

    ################################
    # Create output directory if it does not exist
    ################################
    book_dir = os.path.dirname(book_path)
    # create the chapters subdirectory if it does not exist, chapter_dir is the output directory for the chapters
    chapters_dir = os.path.join(book_dir, 'chapters').replace('\\', '/')
    if not os.path.exists(chapters_dir):
        os.makedirs(chapters_dir)

    ###############################
    # Extract text from the PDF and save to CSV
    ###############################
    print(f"Extracting text from {book_path} from page {start_page} to {end_page}...")

    # Iterrate through the chapter mapping and extract text for each chapter
    for idx, (chapter_name, (start, end)) in tqdm(enumerate(chapter_mapping.items()), total = len(chapter_mapping), desc="Extracting chapters"):
        chapter_output_dir_path = os.path.join(chapters_dir)
        os.makedirs(chapter_output_dir_path, exist_ok=True)
        chapter_output_csv_path = os.path.join(chapter_output_dir_path, f"idx_{idx+1}_{chapter_name}.csv").replace('\\', '/')
        print(f"Extracting text for {chapter_name} from page {start} to {end}...")
        extract_text_from_pdf(book_path, start, end, chapter_output_csv_path)
        print(f"Text extracted and saved to {chapter_output_csv_path}")

    print("Extraction complete.")