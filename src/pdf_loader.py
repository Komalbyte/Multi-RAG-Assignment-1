"""
pdf_loader.py
Handles reading PDFs and pulling out the text.
Nothing fancy here - just pypdf doing its thing.
"""

from pypdf import PdfReader


def load_pdf(file_path):
    """
    Reads a PDF and gets all the text out of it.
    Returns a dict with the text, page count, etc.
    """
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        txt = page.extract_text()
        # some pages might be scanned images, so we just skip those
        pages.append(txt if txt else "")

    full_text = "\n".join(pages)

    info = {
        "text": full_text,
        "num_pages": len(reader.pages),
        "num_chars": len(full_text),
        "pages": pages,
    }

    return info


def show_pdf_info(info):
    """Quick printout to check if the PDF loaded okay."""
    print(f"Pages: {info['num_pages']}")
    print(f"Characters: {info['num_chars']}")
    # just show a bit of the first page so we know it worked
    first_page = info['pages'][0][:300] if info['pages'] else "(empty)"
    print(f"Preview:\n{first_page}...")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <pdf_path>")
        sys.exit(1)

    data = load_pdf(sys.argv[1])
    show_pdf_info(data)
