from cosmos_reader import CosmosReader


class Chunker:

    def __init__(self, chunk_size=800, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_text(self, text):
        """
        Split text into overlapping chunks.
        """

        if not text:
            return []

        chunks = []

        start = 0

        while start < len(text):

            end = min(start + self.chunk_size, len(text))

            chunks.append(text[start:end])

            if end == len(text):
                break

            start = end - self.overlap

        return chunks

    def chunk_document(self, document):

        chunks = []

        chunk_number = 0

        fields = document.get("fields", {})

        if not isinstance(fields, dict):
            return chunks

        for section_name, section in fields.items():

            if not isinstance(section, dict):
                continue

            text = section.get("ai_editable_abstract_content")

            if not text:
                continue

            text_chunks = self.split_text(str(text))

            for piece in text_chunks:

                chunks.append({
                    "chunk_id": f"{document['id']}_{chunk_number}",
                    "document_id": document["id"],
                    "organization_id": document["organization_id"],
                    "section": section_name,
                    "text": piece,

                    # Useful metadata
                    "tenant_name": document.get("tenant_name"),
                    "landlord_name": document.get("landlord_name"),
                    "property_address": document.get("property_address"),
                    "lease_term": document.get("lease_term"),
                    "commencement_date": document.get("commencement_date"),
                    "expiration_date": document.get("expiration_date"),
                    "type": document.get("type")
                })

                chunk_number += 1

        return chunks


def main():

    reader = CosmosReader()
    chunker = Chunker()

    total_chunks = 0

    for organization_id in reader.get_organization_ids():

        print(f"\nProcessing Organization: {organization_id}")

        documents = reader.load_documents_by_organization(
            organization_id
        )

        for document in documents:

            chunks = chunker.chunk_document(document)

            total_chunks += len(chunks)

            print(f"\nDocument: {document['id']}")
            print(f"Generated {len(chunks)} chunks")

            for chunk in chunks[:3]:
                print("-" * 60)
                print(f"Chunk ID : {chunk['chunk_id']}")
                print(f"Section  : {chunk['section']}")
                print(f"Length   : {len(chunk['text'])} characters")
                print(chunk["text"][:250] + "...")
                print()

    print("=" * 60)
    print(f"Total Chunks Generated: {total_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    main()