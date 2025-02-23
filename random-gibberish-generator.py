import subprocess
import random
import os

# Install Faker if not installed
try:
    from faker import Faker
except ImportError:
    subprocess.check_call(["pip", "install", "Faker"])
    from faker import Faker

# Creates Faker generator
fake = Faker()
fake.seed_instance(42)  # FIX: Ensures Faker outputs the same fake words every time
counter = 0
long_strings = []
while counter <= 5000:
    new_one = fake.text()  # Really long string
    long_strings.append(new_one)
    counter += 1

sentences = [item.split('.') for item in long_strings]  # List of lists

# FIX: Changed "a" to "w" to avoid appending indefinitely, ensuring a fresh file each time.
with open("gibberish_sentences.txt", "w", encoding="utf-8") as outputFile:
    for sentence in sentences:
        for phrase in sentence:
            if phrase.strip():  # Avoid empty lines
                cleaned_phrase = phrase.lower().strip(' .\n')
                outputFile.write(cleaned_phrase + '\n')

# Read and combine with "other_ES_clean_.txt"
other_file_path = os.path.join("Jessica's part", "other_ES_clean_.txt")

if os.path.exists(other_file_path):  # FIX: Check if file exists before reading it
    with open(other_file_path, "r", encoding="utf-8") as other_ES_clean_txt, \
         open("gibberish_sentences.txt", "r", encoding="utf-8") as gibberishFile, \
         open("combined_gibberish_english.txt", "w", encoding="utf-8") as finalFile:
        
        finalFile.write(gibberishFile.read())
        finalFile.write(other_ES_clean_txt.read())
else:
    print(f"Error: {other_file_path} not found!")

#--- Breaks up large file into smaller files ---
'''with open ("combined_gibberish_english.txt","r") as file:
    data_1to8000 = []
    counter = 0
    while counter <=8000:
        data_1to8000.append(file.readline())
        counter+=1
with open ("f1.txt", "w") as outfile:
    outfile.write(''.join(data_1to8000))'''

def split_large_file(input_file, lines_per_file=8000):
    """
    Splits a large text file into smaller files, each containing a specified number of lines.

    :param input_file: The path to the large file to be split.
    :param lines_per_file: The number of lines per smaller file.
    """
    file_count = 1  # Start output file numbering
    output_dir = "split_files"  # Folder for split files
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    with open(input_file, "r", encoding="utf-8") as file:
        while True:
            data_chunk = [file.readline() for _ in range(lines_per_file)]
            data_chunk = [line for line in data_chunk if line.strip()]  # Remove empty lines

            if not data_chunk:  # Stop when no more data
                print(f"Finished splitting. Created {file_count - 1} files.")
                break

            output_file = os.path.join(output_dir, f"f{file_count}.txt")
            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.writelines(data_chunk)

            print(f"Created: {output_file} ({len(data_chunk)} lines)")
            file_count += 1
