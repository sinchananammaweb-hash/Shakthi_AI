import json

with open("datasets/government_data/government_data.json", "r", encoding="utf-8") as f:
    government_data = json.load(f)

with open("datasets/government_data/test_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

print("Government records:", len(government_data))
print("QA pairs:", len(qa_data))

government_ids = {record["id"] for record in government_data}

valid = 0
invalid = 0

for qa in qa_data:
    if qa["source_id"] in government_ids:
        valid += 1
    else:
        invalid += 1
        print("Invalid source_id:", qa["source_id"])

print("\nQA source validation:")
print("Valid:", valid)
print("Invalid:", invalid)

if invalid == 0:
    print("\nAll QA pairs are correctly linked to government records.")
else:
    print("\nSome QA pairs have invalid source IDs.")
