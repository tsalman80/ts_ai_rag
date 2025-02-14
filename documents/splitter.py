from langchain_text_splitters import CharacterTextSplitter


class DocumentSplitter:
    """
    Splits the documents into chunks of text.
    """

    @staticmethod
    def chunk(documents):
        """
            Splits the documents into chunks of text.

        Args:
            documents: List of loaded document objects
        Returns:
            chunks: List of chunks of text
        """

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(documents)
        return chunks
