from langchain_text_splitters import CharacterTextSplitter

# 1. Read the text file
with open("../data/notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Original text:")
print("-" * 40)
print(text)

# 2. Create a text splitter
text_splitter = CharacterTextSplitter(
    separator="\n\n",   # try to split on blank lines (paragraphs)
    chunk_size=150,     # target size of each chunk (characters)
    chunk_overlap=30,   # how much chunks overlap
    length_function=len
)

# 3. Split into chunks
chunks = text_splitter.split_text(text)

print("\n\nChunks:")
print("=" * 40)
for i, chunk in enumerate(chunks, start=1):
    print(f"[Chunk {i}]")
    print(chunk)
    print("-" * 40)