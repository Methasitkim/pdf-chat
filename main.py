import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

def load_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def split_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def build_vectordb(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=True)
    client = chromadb.Client()
    collection = client.create_collection("pdf_docs")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def ask(question, collection, embed_model):
    # แปลงคำถามเป็น embedding
    q_embedding = embed_model.encode([question]).tolist()

    # หา chunks ที่เกี่ยวข้องที่สุด 3 อัน
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=3
    )
    context = "\n\n".join(results["documents"][0])

    # ส่งให้ AI ตอบ
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")

    response = model.generate_content(f"""ตอบคำถามโดยอ้างอิงจากเนื้อหาที่ให้มาเท่านั้น
ถ้าไม่มีข้อมูลในเนื้อหา ให้บอกว่า "ไม่พบข้อมูลในเอกสาร"

เนื้อหา:
{context}

คำถาม: {question}""")

    return response.text

if __name__ == "__main__":
    path = "test.pdf"

    print("กำลังเตรียมระบบ...")
    text = load_pdf(path)
    chunks = split_chunks(text)

    print("กำลังโหลด embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("กำลังสร้าง vector database...")
    collection = build_vectordb(chunks, embed_model)
    print(f"✅ พร้อมแล้ว! ({collection.count()} chunks)\n")

    # โหมด chat
    print("💬 ถามคำถามเกี่ยวกับ PDF ได้เลย (พิมพ์ 'exit' เพื่อออก)\n")
    while True:
        question = input("คำถาม: ")
        if question.lower() == "exit":
            break
        answer = ask(question, collection, embed_model)
        print(f"\nคำตอบ: {answer}\n")
        print("-" * 50)