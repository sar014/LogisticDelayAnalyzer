import csv
with open('C:\\Users\\hp\\Desktop\\AIDTM\\GenAI\\End-TermProject\\Final_Code\\smart_logistics_dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row, "\n")
