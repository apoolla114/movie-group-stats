from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from collections import Counter
import time

app = Flask(__name__)
CORS(app)

BASE_URL = "https://letterboxd.com"

def extract_title_and_year(full_name):
    """Extract title and year from 'Title (Year)' format"""
    title = full_name
    year = None
    
    if '(' in full_name and ')' in full_name:
        parts = full_name.rsplit('(', 1)
        potential_year = parts[1].rstrip(')').strip()
        
        if potential_year.isdigit() and len(potential_year) == 4:
            title = parts[0].strip()
            year = potential_year
    
    return title, year

def scrape_user_films(username):
    """Scrape films from a user using slug + data-item-name"""
    films = []
    page = 1
    max_pages = 3  # Reduced for speed
    
    print(f"Scraping films for user: {username}")
    
    while page <= max_pages:
        try:
            url = f"{BASE_URL}/{username}/films/page/{page}/"
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0'
            })
            
            if response.status_code != 200:
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            film_items = soup.select('li.griditem')
            
            if not film_items:
                break
            
            for item in film_items:
                try:
                    react_div = item.select_one('div.react-component')
                    if not react_div:
                        continue
                    
                    # Get slug
                    film_slug = react_div.get('data-target-link', '')
                    if not film_slug:
                        continue
                    
                    # Get full name
                    full_name = react_div.get('data-item-name', '')
                    if not full_name:
                        continue
                    
                    # Parse title and year
                    title, year = extract_title_and_year(full_name)
                    
                    films.append({
                        'slug': film_slug,
                        'title': title,
                        'year': year
                    })
                    
                except Exception as e:
                    continue
            
            page += 1
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            break
    
    print(f"Found {len(films)} films for {username}")
    return films

@app.route('/api/analyze', methods=['POST'])
def analyze_group():
    """Analyze group - simple version"""
    try:
        data = request.get_json()
        usernames = data.get('usernames', [])
        
        if len(usernames) < 2:
            return jsonify({'error': 'At least 2 usernames required'}), 400
        
        print(f"Analyzing: {usernames}")
        
        # Scrape films for each user
        user_films = {}
        for username in usernames:
            films = scrape_user_films(username)
            if not films:
                return jsonify({'error': f'Could not find user: {username}'}), 404
            user_films[username] = films
        
        # Find shared films (by slug)
        all_slugs = [set(f['slug'] for f in films) for films in user_films.values()]
        shared_slugs = set.intersection(*all_slugs)
        
        # Get shared film details
        shared_films = []
        for slug in shared_slugs:
            # Find this film in any user's list
            for films in user_films.values():
                for film in films:
                    if film['slug'] == slug:
                        shared_films.append({
                            'title': film['title'],
                            'year': film['year']
                        })
                        break
                break
        
        # Calculate stats
        total_films = sum(len(films) for films in user_films.values())
        
        # Count years
        all_years = [f['year'] for films in user_films.values() 
                     for f in films if f['year']]
        year_counts = Counter(all_years)
        top_years = [{'year': year, 'count': count} 
                     for year, count in year_counts.most_common(5)]
        
        # Response
        response = {
            'user_count': len(usernames),
            'total_films': total_films,
            'shared_films': shared_films[:20],  # Limit to 20
            'top_years': top_years
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("Starting Simple Letterboxd API...")
    print("Install: pip install flask flask-cors requests beautifulsoup4")
    app.run(debug=True, host='0.0.0.0', port=5000)