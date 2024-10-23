import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to activate the env source venv/bin/activate

def test_classifier():
    """
    Test the news classifier API with functional and performance tests
    """
    base_url = 'http://serve-sentiment-env.eba-gvdinfyp.us-east-2.elasticbeanstalk.com'
    
    # Test cases - 2 fake news and 2 real news examples
    test_cases = {
        'fake_news_1': 'This shocking news will blow your mind! Scientists discover dragons are real!',
        'fake_news_2': 'Secret government program reveals aliens living among us - MUST READ!',
        'real_news_1': 'The Federal Reserve announced a quarter-point increase in interest rates today.',
        'real_news_2': 'NASA successfully launched its new Mars rover mission this morning from Cape Canaveral.'
    }
    
    # Part 1: Functional Tests
    print("\nPart 1: Running Functional Tests")
    print("---------------------------------")
    
    for name, text in test_cases.items():
        response = requests.post(
            f"{base_url}/predict",
            json={'text': text}
        )
        result = response.json()
        print(f"\nTest Case: {name}")
        print(f"Text: {text}")
        print(f"Prediction: {result['prediction']}")
        print(f"Processing Time: {result['processing_time']:.3f} seconds")
    
    # Part 2: Performance Tests
    print("\nPart 2: Running Performance Tests")
    print("---------------------------------")
    
    # Store results for each test case
    all_results = []
    
    # Run 100 API calls for each test case
    for name, text in test_cases.items():
        print(f"\nRunning 100 calls for: {name}")
        
        for i in range(100):
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/predict",
                json={'text': text}
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            all_results.append({
                'test_case': name,
                'latency': latency,
                'iteration': i + 1
            })
            
            if (i + 1) % 20 == 0:  # Progress update every 20 calls
                print(f"Completed {i + 1} calls")
    
    # Create DataFrame with results
    df = pd.DataFrame(all_results)
    
    # Save results to CSV
    df.to_csv('api_test_results.csv', index=False)
    print("\nResults saved to api_test_results.csv")
    
    # Calculate average performance
    avg_latency = df.groupby('test_case')['latency'].mean()
    print("\nAverage Latency per Test Case:")
    print(avg_latency)
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='test_case', y='latency', data=df)
    plt.xticks(rotation=45)
    plt.title('API Latency Distribution by Test Case')
    plt.xlabel('Test Case')
    plt.ylabel('Latency (seconds)')
    plt.tight_layout()
    plt.savefig('latency_boxplot.png')
    print("\nBoxplot saved as latency_boxplot.png")

if __name__ == "__main__":
    test_classifier()