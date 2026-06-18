import urllib.request
import urllib.error
import csv
import json
import sys

def main():
    url = "https://jsonplaceholder.typicode.com/posts"
    timeout = 10
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.getcode() != 200:
                print(f"HTTP error: {response.getcode()}", file=sys.stderr)
                return
            
            data = response.read().decode('utf-8')
            posts = json.loads(data)
            
            with open('posts.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'userId', 'title'])
                for post in posts:
                    writer.writerow([post['id'], post['userId'], post['title']])
                    
    except urllib.error.URLError as e:
        print(f"URL error: {e}", file=sys.stderr)
        return
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return

if __name__ == "__main__":
    main()
