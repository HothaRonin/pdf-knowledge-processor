import os
import pdfplumber

INPUT_DIR = "input_pdfs"
OUTPUT_DIR = "output_texts"
LOG_FILE = "logs/extraction_log.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_entries = []

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".pdf"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".pdf", ".txt"))

        try:
            with pdfplumber.open(input_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            if text.strip():
                with open(output_path, "w", encoding="utf-8") as out_file:
                    out_file.write(f"=== Begin: {filename} ===\n{text}\n=== End: {filename} ===")
                log_entries.append(f"{filename}: Success")
            else:
                log_entries.append(f"{filename}: No extractable text")

        except Exception as e:
            log_entries.append(f"{filename}: Failed with error {e}")

with open(LOG_FILE, "w") as log:
    for entry in log_entries:
        log.write(entry + "\n")

print("Extraction complete. See logs/extraction_log.txt for details.")
