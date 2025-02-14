import os
from typing import BinaryIO
from config import MAX_FILE_SIZE


class DocumentProcessor:
    @staticmethod
    def is_valid_file_size(content: BinaryIO) -> bool:
        """Validate the file size"""

        try:
            content.seek(0, os.SEEK_END)
            file_size = content.tell()
            content.seek(0)
        except Exception as e:
            print(f"Error validating file size: {e}")
            return False

        print(f"File size: {file_size} - {MAX_FILE_SIZE}")
        if file_size > MAX_FILE_SIZE:
            return False

        return True
