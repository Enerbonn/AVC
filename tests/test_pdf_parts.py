import tempfile
import unittest
from pathlib import Path
from reportlab.pdfgen import canvas
from pypdf import PdfReader
from src.pdf_parts import split_pdf


class PDFPartTests(unittest.TestCase):
    def test_205_pages_split_into_numbered_parts_with_content_and_timestamps_intact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "master.pdf"
            doc = canvas.Canvas(str(original))
            for number in range(205):
                doc.drawString(12, 800, f"Video timestamp: 00:{number//60:02d}:{number%60:02d}.000")
                doc.drawString(12, 780, f"Original page {number+1}")
                doc.showPage()
            doc.save()
            parts = split_pdf(original, root / "parts", "lesson")
            self.assertEqual([Path(p["file"]).name for p in parts], ["lesson_01.pdf", "lesson_02.pdf", "lesson_03.pdf"])
            self.assertEqual([p["pages"] for p in parts], [100, 100, 5])
            before = [p.extract_text() for p in PdfReader(original).pages]
            after = [p.extract_text() for part in parts for p in PdfReader(part["file"]).pages]
            self.assertEqual(after, before)
            with self.assertRaises(ValueError):
                split_pdf(original, root / "parts", "lesson")

    def test_small_pdf_keeps_video_name(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); doc = canvas.Canvas(str(root / "master.pdf"));doc.drawString(10,10,"timestamp");doc.save()
            result = split_pdf(root / "master.pdf", root / "parts", "recording")
            self.assertEqual(Path(result[0]["file"]).name, "recording.pdf")


if __name__ == "__main__":
    unittest.main()
