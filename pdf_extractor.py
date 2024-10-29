import pdfplumber
import json
import os

# Folder paths
pdf_folder = 'pdf_folder'   # Replace with your folder path containing PDF files
output_json_file = 'dataset.json'

# Sample data structure
data = []

# Define instructions and expected output placeholders (for demonstration)
instructions = "Extract the Invoice No, Invoice Date, and SS Name."
sample_outputs = [
    "Invoice No: 58/24-25, Invoice Date: 26-Jun-24, SS Name: JAI ENTERPRISES.",
    "Invoice No: 59/24-25, Invoice Date: 27-Jun-24, SS Name: ABC COMPANY."
]

# Helper function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text_content = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n"  # Add newline between pages
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text_content

# Iterate through PDF files in the folder
for idx, pdf_file in enumerate(os.listdir(pdf_folder)):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pdf_content = extract_text_from_pdf(pdf_path)
        
        # Create a structured dictionary for each PDF file
        pdf_data = {
            "instruction": instructions,
            "input": pdf_content,
            "output": sample_outputs[idx % len(sample_outputs)]  # Use sample output in rotation
        }
        
        # Append to the data list
        data.append(pdf_data)

# Write the data to a JSON file
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f'Dataset saved to {output_json_file}')
