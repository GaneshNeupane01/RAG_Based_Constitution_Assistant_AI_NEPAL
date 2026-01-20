from docling.document_converter import DocumentConverter
import json

converter = DocumentConverter()

source = "Constitution.pdf"

print(f"Converting {source} ... this might take a minute as it analyzes layout.")

result = converter.convert(source)


markdown_content = result.document.export_to_markdown()


json_content = result.document.export_to_dict()

# Save the outputs
with open("converted_law.md", "w") as f:
    f.write(markdown_content)

print("--- CONVERSION SUCCESSFUL ---")
print(markdown_content[:500])