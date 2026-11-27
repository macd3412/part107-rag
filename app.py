from src.rag.pipeline import Part107RAG

def main() -> None:
    rag = Part107RAG()
    print("Part 107 Drone FAQ RAG ready! Type 'quit' to exit.\n")
    while True:
        question = input("Ask a Part 107 question: ").strip()
        if not question or question.lower() in {"quit", "exit", "q"}:
            break
        answer, sources = rag.ask(question)
        print("\n" + answer + "\n")
        print("Retrieved sections:")
        for s in sources:
            print(f"  - {s['citation']}: {s['section_title']}")
        print()

if __name__ == "__main__":
    main()
