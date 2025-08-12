class Author:
    def __init__(self, id: int, name: str, details: str = None):
        self.id = id
        self.name = name
        self.details = details

    def __str__(self):
        return f"Author(id={self.id}, name='{self.name}', details='{self.details}')"

    def __repr__(self):
        return self.__str__()
