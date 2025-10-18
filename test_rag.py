import requests
import time
from typing import Dict, Any

BASE_URL = "http://localhost:5001"

def test_health():
    """Test the health endpoint."""
    print("Testing health endpoint")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"Health check passed: {data['status']}")
            print(f"   - Index loaded: {data['index_loaded']}")
            print(f"   - Chunks available: {data['chunks_available']}")
            print(f"   - Ollama status: {data['ollama_status']}")
            print(f"   - Model: {data['model']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_question(question: str, expected_keywords: list = None) -> Dict[str, Any]:
    """Test asking a question and return the response."""
    print(f"\nQuestion: {question}")
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/ask",
            json={"prompt": question},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            processing_time = time.time() - start_time
            
            print(f"Response received in {processing_time:.2f}s")
            print(f"   - Processing time: {data['metadata']['processing_time']:.2f}s")
            print(f"   - Sources used: {data['metadata']['sources_used']}")
            print(f"   - Model: {data['metadata']['model_used']}")

            if expected_keywords:
                response_text = data['response'].lower()
                found_keywords = [kw for kw in expected_keywords if kw.lower() in response_text]
                if found_keywords:
                    print(f"   - Found expected keywords: {found_keywords}")
                else:
                    print(f"   - Expected keywords not found: {expected_keywords}")
            
            response_preview = data['response'][:200] + "..." if len(data['response']) > 200 else data['response']
            print(f"   - Response preview: {response_preview}")
            
            return data
        else:
            print(f"Request failed: {response.status_code}")
            print(f"   - Error: {response.text}")
            return {}
            
    except Exception as e:
        print(f"Request error: {e}")
        return {}

def test_stats():
    """Test the stats endpoint."""
    print("\nTesting stats endpoint")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"Stats retrieved:")
            print(f"   - Total pages: {data['total_pages']}")
            print(f"   - Total chunks: {data['total_chunks']}")
            print(f"   - Total characters: {data['total_characters']}")
            print(f"   - Average chunk size: {data['avg_chunk_size']:.0f}")
            return True
        else:
            print(f"Stats request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Stats error: {e}")
        return False

def test_sources():
    """Test the sources endpoint."""
    print("\n📚 Testing sources endpoint")
    try:
        response = requests.get(f"{BASE_URL}/sources")
        if response.status_code == 200:
            data = response.json()
            print(f"Sources retrieved: {len(data['sources'])} sources")
            for source in data['sources']:
                print(f"   - {source}")
            return True
        else:
            print(f"Sources request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Sources error: {e}")
        return False

def main():
    """Comprehensive tests of the RAG"""
    print("Starting Eluvio RAG System Tests")
    print("=" * 50)
    if not test_health():
        print("Health check failed. Make sure the server is running.")
        return
    
    test_cases = [
        {
            "question": "What does Eluvio do?",
            "expected_keywords": ["Content Fabric", "streaming", "video", "platform"]
        },
        {
            "question": "What technology does Eluvio use?",
            "expected_keywords": ["blockchain", "AI", "edge computing", "decentralized"]
        },
        {
            "question": "How does the Content Fabric work?",
            "expected_keywords": ["decentralized", "network", "distribution", "creators"]
        },
        {
            "question": "What are Eluvio's key features?",
            "expected_keywords": ["streaming", "storage", "blockchain", "APIs"]
        },
        {
            "question": "Who does Eluvio serve?",
            "expected_keywords": ["media companies", "content creators", "enterprises"]
        }
    ]
    print("\nTesting various questions")
    successful_tests = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}/{len(test_cases)} ---")
        result = test_question(
            test_case["question"], 
            test_case["expected_keywords"]
        )
        if result:
            successful_tests += 1
    test_stats()
    test_sources()

    print("\n" + "=" * 50)
    print("Test Summary")
    print(f"Successful question tests: {successful_tests}/{len(test_cases)}")
    print(f"Health check: {'PASSED' if test_health() else 'FAILED'}")
    print(f"Stats endpoint: {'PASSED' if test_stats() else 'FAILED'}")
    print(f"Sources endpoint: {'PASSED' if test_sources() else 'FAILED'}")
    
    if successful_tests == len(test_cases):
        print("\nAll tests passed! The RAG system is working perfectly.")
    else:
        print(f"\n{len(test_cases) - successful_tests} tests failed. Check the server logs.")

if __name__ == "__main__":
    main()
