from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader


class DocumentLoader:
    """
    Loads PDF documents from the temporary directory.
    """

    @staticmethod
    def load(dir_path):
        """
        Loads PDF documents from the temporary directory.

        Returns:
            documents: List of loaded document objects
        """

        loader = PyPDFDirectoryLoader(dir_path, glob="**/*.pdf")
        documents = loader.load()
        return documents
