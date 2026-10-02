import json

text = open("input.txt", encoding="utf-8").read()
pezzi = []

for i in range(0, len(text), 400):
    pezzo = text[i:i+400].strip()
    if len(pezzo) > 50:
        pezzi.append(pezzo)

with open("pretrain_data.jsonl", "w", encoding="utf-8") as f:
    for p in pezzi:
        f.write(json.dumps({"text": p}) + "\n")

print("Fatto:", len(pezzi), "righe scritte")