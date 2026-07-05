from rag_core import load_and_chunk, build_vectorstore, get_chain
import json

# ── Test set — question + expected keywords in answer ──
TEST_SET = [
    {
        "question": "What is the time complexity of binary search?",
        "expected_keywords": ["log n", "O(log n)", "logarithmic"],
        "subject": "DSA"
    },
    {
        "question": "What is polymorphism?",
        "expected_keywords": ["overloading", "overriding", "many forms"],
        "subject": "OOPS"
    },
    {
        "question": "What are ACID properties?",
        "expected_keywords": ["atomicity", "consistency", "isolation", "durability"],
        "subject": "DBMS"
    },
    # {
    #     "question": "What is a deadlock?",
    #     "expected_keywords": ["wait", "circular", "resource", "process"],
    #     "subject": "OS"
    # },
    # {
    #     "question": "What is DNS?",
    #     "expected_keywords": ["domain", "IP", "name resolution"],
    #     "subject": "CN"
    # },
]

def evaluate(pdf_files):
    chunks = load_and_chunk(pdf_files)
    vectorstore = build_vectorstore(chunks, force_rebuild=True)

    results = []
    passed = 0

    for test in TEST_SET:
        chain = get_chain(vectorstore, subject=test["subject"])
        response = chain.invoke({"query": test["question"]})
        answer = response.get("result", response.get("answer", "")).lower()

        # Check if expected keywords appear in answer
        hits = [
            kw for kw in test["expected_keywords"]
            if kw.lower() in answer
        ]
        score = len(hits) / len(test["expected_keywords"])
        status = "PASS" if score >= 0.5 else "❌FAIL"
        if score >= 0.5:
            passed += 1

        results.append({
            "question": test["question"],
            "subject": test["subject"],
            "answer": response.get("result",response.get("answer", "")),
            "score": f"{score:.0%}",
            "status": status
        })

        print(f"{status} [{test['subject']}] {test['question']}")
        print(f"   Score: {score:.0%} | Hits: {hits}\n")

    print(f"\n📊 Overall: {passed}/{len(TEST_SET)} passed")
    
    # Save results
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("💾 Results saved to eval_results.json")

if __name__ == "__main__":
    PDF_FILES = [
            {"path": "data/dsa_notes.pdf", "name": "dsa_notes.pdf"},
            {"path": "data/oops_notes.pdf", "name": "oops_notes.pdf"},
            {"path": "data/dbms_notes.pdf", "name": "dbms_notes.pdf"},
            # {"path": "data/os_notes.pdf", "name": "os_notes.pdf"},
            # {"path": "data/cn_notes.pdf", "name": "cn_notes.pdf"},
    ]
    evaluate(PDF_FILES)