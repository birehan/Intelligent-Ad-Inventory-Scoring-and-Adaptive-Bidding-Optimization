from docx import Document
import fitz
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DataExtractor:
    """
    A class for extracting text data from various file formats.
    """

    @staticmethod
    def extract_data(file_path: str) -> str:
        """
        Extract text data from different file formats.

        Args:
            file_path (str): The path to the file to extract text data from.

        Returns:
            str: The extracted text data.
        """
        try:
            # Get file extension
            file_ext = file_path.split('.')[-1].lower()
            # Map file extensions to extraction methods
            extraction_methods = {
                'pdf': DataExtractor.extract_pdf_data,
                'docx': DataExtractor.extract_docx_data,
                'txt': DataExtractor.extract_txt_data
            }
            # Extract data based on file extension
            if file_ext in extraction_methods:
                return extraction_methods[file_ext](file_path)
            else:
                raise Exception("Unsupported file type")
        except Exception as e:
            logging.error(f"Error extracting data: {e}")
            return ""

    @staticmethod
    def extract_pdf_data(file_path: str) -> str:
        """
        Extract text data from a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text data.
        """
        try:
            pdf_data = ""
            with fitz.open(file_path) as pdf_file:
                for page_num in range(len(pdf_file)):
                    page = pdf_file.load_page(page_num)
                    pdf_data += page.get_text()
            return pdf_data
        except Exception as e:
            logging.error(f"Error extracting PDF data: {e}")
            return ""

    @staticmethod
    def extract_docx_data(file_path: str) -> str:
        """
        Extract text data from a DOCX file.

        Args:
            file_path (str): The path to the DOCX file.

        Returns:
            str: The extracted text data.
        """
        try:
            docx_data = ""
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                docx_data += paragraph.text + "\n"
            return docx_data
        except Exception as e:
            logging.error(f"Error extracting DOCX data: {e}")
            return ""

    @staticmethod
    def extract_txt_data(file_path: str) -> str:
        """
        Extract text data from a TXT file.

        Args:
            file_path (str): The path to the TXT file.

        Returns:
            str: The extracted text data.
        """
        try:
            with open(file_path, 'r') as txt_file:
                txt_data = txt_file.read()
            return txt_data
        except Exception as e:
            logging.error(f"Error extracting TXT data: {e}")
            return ""
