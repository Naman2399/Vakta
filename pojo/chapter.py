from typing import List

from pojo.dialogue import Dialogue


class Chapter:
    def __init__(self, id: int, title: str, dialogues: List[Dialogue]):
        self.id = id
        self.title = title
        self.dialogues = dialogues

    def __str__(self):
        return f"Chapter(id={self.id}, title='{self.title}', dialogues='{self.dialogues}')"

    def __repr__(self):
        return self.__str__()