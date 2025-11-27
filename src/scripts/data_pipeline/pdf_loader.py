from langchain_community.document_loaders import PyPDFLoader, TextLoader


class PDFLoader:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.pdf_loader = PyPDFLoader(file_path)
        self.txt_loader = TextLoader(file_path)

    def load_document(self) -> str:
        loader = self.pdf_loader if self.file_path.endswith(".pdf") else self.txt_loader
        return "".join([page.page_content for page in loader.load()])
