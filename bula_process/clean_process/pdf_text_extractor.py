import PyPDF2

class PDFTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
    
    def extract_text(self):
        text = ""
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages) - 1
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()

        # Split the text into lines and filter out empty lines
        lines = filter(lambda x: x.strip(), text.split('\n'))
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text
