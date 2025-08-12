from pojo.author import Author
from typing import List

from pojo.chapter import Chapter


class Book :

    def __init(self, id: int, title: str, authors : List[Author], chapters : List[Chapter], references: str, isbn: str = None, publisher: str = None, year: int = None):
        self.id = id
        self.title = title
        self.authors = authors
        self.chapters = chapters
        self.references = references
        self.isbn = isbn
        self.publisher = publisher
        self.year = year

    def __str__(self):
        return f"Book(id={self.id}, title='{self.title}', authors={self.authors}, chapters={self.chapters}, references='{self.references}', isbn='{self.isbn}', publisher='{self.publisher}', year={self.year})"

    def __repr__(self):
        return self.__str__()
