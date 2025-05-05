# PDF Knowledge Processor

This tool extracts clean text from PDFs for use in LLMs, journaling systems, or knowledge bases.

## Features

- Extracts all readable text using `pdfplumber`
- Wraps content in start/end markers per document
- Logs extraction success or failure
- Designed for use in AI workflows or reflection tools

## Usage

1. Place PDFs into the `input_pdfs/` folder
2. Run `pdf_to_text.py`
3. Outputs will appear in `output_texts/`
4. Check `logs/extraction_log.txt` for a summary

## Dependencies

Install with:

```bash
pip install -r requirements.txt
```

## Notes

- This version does **not** use OCR. It only works with PDFs that contain embedded text.
- Images, figures, and charts are ignored.
