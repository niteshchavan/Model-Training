from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
import subprocess

# Load the fine-tuned model and tokenizer
#model, tokenizer = FastLanguageModel.from_pretrained("lora_model")

power = subprocess.run(
                ["upower", "-i", "/org/freedesktop/UPower/devices/battery_BAT1"],
            capture_output=True,
            text=True,
            check=True
        )

poutput = power.stdout
print(poutput)
model, tokenizer = FastLanguageModel.from_pretrained("lora_model", device_map="cuda")

major_version, minor_version = torch.cuda.get_device_capability()

# Ensure the model is ready for inference
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

# Define your question
instruction = "Extract the Invoice No, Invoice Date, and SS Name."  # e.g., "Translate the following English sentence to French."
input_text = "ORIGINAL\nTAX INVOICE\nGIRIRAJ MARKETING (AHM)\nG.FLOR. MILKAT NO 5249,S.R.ESTATE, B/H. HP WREHOUSE, NEAR BAJRANG ESTATE N H- NO. - 8, ASALALI\nAhmedabad Gujrat 382427 Mob. 9913581068 & 6352357458\nGSTIN No. : 24AALFG6521D1ZP 20211018102899102 PAN No. : AALFG6521D\nTime : 12:36PM :\nInvoice No. : PS-24/25-0457\nInvoice Date : 01/08/2024 Reverse Charge : N/A\nState : Gujarat State Code : 24\nDetails of Receiver (Billed to) Details of Consignee(Shipped to)\nName : Manibhadra Agency (Kalupur) Name : Manibhadra Agency (Kalupur)\nAddress : 2318 Behind Satyanarayan Chokatha Bhanderi Polo Address : 2318 Behind Satyanarayan Chokatha Bhanderi Polo\nDelivary :- D-5 City Centre Near Id Gah Circle Kalupur Delivary :- D-5 City Centre Near Id Gah Circle Kalupu\nMr. Mayurbhai :- 9033001300 Mr. Mayurbhai :- 9033001300\nAhmedabad Ahmedabad\nState : Gujarat State Code : 24 State : Gujarat State Code 2:4\nGSTIN No. : 24APPPJ6190Q1Z2 GSTIN No. : 24APPPJ6190Q1Z2\nPAN No. : APPPJ6190Q PAN No. : APPPJ6190Q\nSr. Description of Goods HSN /SAC Qty Free Unit Rate Disc.% Disc. Sch . % SchAmt. GST % SGST CGST Amount\nQty\n1 PT.Rooperi 50Ml (12X8=96) Mrp.110 34059090 96.00 Pic 63.63 1.25 76.36 0.50 30.16 18.00 540.18 540.18 7082.31\nSub Total 96.000 76.36 30.16 540.18 540.18 7082.31\nGST Summary Amount SGST CGST Total\nSales 3 %\nSales 5 %\nSales 12 %\nSales 18 % 6001.96 540.18 540.18 1080.36 0.00\nSales 28 %\nSales 0 % Round Off -0.32\nTotal 6001.96 540.18 540.18 1080.36 Invoice Total 7082.00\nInvoice Value (In Words) : Seven Thousand Eighty Two Only\nOur Bank Detail : A/c No. : 50200094104733\nBank Name : HDFC BANK NEFT / IFS Code : HDFC0000379\nLast Balance : 499638.00 Debit Current Amt : 7082.00 Net Balance : 506720.00 Debit\nTERM & CONDITION For, GIRIRAJ MARKETING (AHM)\nSubject to Ahmadabad jurisdiction.\nInterest will be charges@24%P.A. on overdue bills.\nGoods once sold will not be taken back or replaced.\nWe are not liable for any cash transaction for our sales staff.\nE. & O. E. Authorised Signatory\nCertified that the particulars given above are true and correct\n"

# Format the prompt for your question
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

question_prompt = alpaca_prompt.format(instruction, input_text, "")

# Tokenize the question prompt
inputs = tokenizer([question_prompt], return_tensors="pt").to("cuda")

# Generate the response
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=1000)

# Print the generated response
#print(output)
