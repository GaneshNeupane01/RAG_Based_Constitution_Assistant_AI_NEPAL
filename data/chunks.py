from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

def split_markdown_into_chunks(
    markdown_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators=None,
    strip_headers: bool = False
):
    """
    Load a markdown file, split by headers, then recursively split into final chunks.

    Returns:
        List[Document]: Final chunked LangChain Document objects
    """

    # Defaults
    if headers_to_split_on is None:
        headers_to_split_on = [
            ("#", "Part"),
        ]

    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    # 1. Load markdown
    with open(markdown_path, "r", encoding="utf-8") as f:
        raw_markdown = f.read()

    # 2. Split by markdown headers
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=strip_headers
    )
    md_header_splits = markdown_splitter.split_text(raw_markdown)

    # 3. Recursive splitting into final chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    final_chunks = text_splitter.split_documents(md_header_splits)

    # Optional verification logs
    print(f"✅ Created {len(md_header_splits)} chunks by markdown headers.")
    print(f"✅ Created {len(final_chunks)} final chunks.")

    return final_chunks
