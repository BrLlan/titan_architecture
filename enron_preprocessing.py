import json
import re

def parse_jsonl_and_process_sentences(jsonl_text):
    # Parse the JSONL string into a list of dictionaries
    data = [json.loads(line) for line in jsonl_text.strip().split('\n') if line]

    # Regex pattern to match periods that likely end sentences. Does not match decimal numbers or single letters (like initials).
    sentence_end_pattern = r"(?<!\w\.\w.)(?<=\.|\?|\!)\s"

    # Function to split text into sentences using regex pattern
    def split_into_sentences(text):
        return [sentence.strip() for sentence in re.split(sentence_end_pattern, text) if sentence.strip()]

    # Iterate through each entry and process the text
    results = []
    for entry in data:
        text = entry.get("text", "")
        sentences = split_into_sentences(text)
        
        # Pairing sentences
        for i in range(len(sentences) - 1):
            results.append((sentences[i], sentences[i + 1]))

    return results

# Example usage with your provided sample
jsonl_text = """{"text":"Kim,\\n\\nAgain, to reiterate my changes, include the following general terms of your\\noffer in the LOU.\\n- Seller to provide all incremental gas needs up to 6,197 MMBtu\\/day at Malin\\npriced at bidweek flat and remaining incremental gas needs at PG&E citygate\\nbidweek flat.\\n- Seller to act as the Buyer's imbalance agent and provide such services as\\nnomination, scheduling and operations for all gas delivered to Buyer,\\nincluding gas purchased from a third party supplier.\\n- Seller to protect Buyer from daily price risk and penalties on EFO\\/OFO\\nevents.\\n- Seller to provide in an electronic format daily price indications.\\n\\nAlso, I spoke to Karla about this and she too had some changes.  First,\\nchange the term \\"demand\\" charge to volumetric charge, since demand charge is\\ncommonly assessed on nominated not actual load.  And change the deadline to\\nJune 30, 2001 (5 days can make a difference).\\n\\nThanks,\\n\\nMonica\\n\\n-----Original Message-----\\nFrom: Ward, Kim S. [mailto:Kim.Ward@ENRON.com]\\nSent: Tuesday, May 15, 2001 1:52 PM\\nTo: Monica Padilla (E-mail)\\nSubject: Letter of Understanding\\n\\n\\nMonica,\\n\\n      Attached is a copy of the letter of understanding.  Barry will be\\nsigning the originals and we will overnight them to you tonight.  Let me\\nknow if you need anything else and I will send you prices tomarrow.\\n\\nThanks,\\n\\nKim\\n\\n <<Letterofunderstanding1.DOC>> "}"""

pairs = parse_jsonl_and_process_sentences(jsonl_text)
for pair in pairs:
    print(f"Previous: {pair[0]}, Following: {pair[1]}")