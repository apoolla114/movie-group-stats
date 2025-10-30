from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from collections import Counter
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict
import threading
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

BASE_URL = "https://letterboxd.com"

# ============= DATA CLASSES =============

@dataclass
class Film:
    """Data class for a film"""
    slug: str
    title: str
    year: str = None
    genres: List[str] = None
    director: str = None
    user_rating: float = None

# ============= ML HELPER FUNCTIONS =============

def create_user_vectors(user_films: Dict[str, List[Film]]) -> Dict:
    """Create feature vectors for each user"""
    print("create_user_vectors called")
    all_genres = set()
    
    # Collect all genres
    for films in user_films.values():
        for film in films:
            if film.genres:
                all_genres.update(film.genres)
    
    if not all_genres:
        return {}
    
    all_genres = sorted(list(all_genres))
    user_vectors = {}
    
    for username, films in user_films.items():
        # Genre counts
        genre_counts = Counter()
        ratings = []
        years = []
        
        for film in films:
            if film.genres:
                for genre in film.genres:
                    genre_counts[genre] += 1
            if film.user_rating:
                ratings.append(film.user_rating)
            if film.year and film.year.isdigit():
                years.append(int(film.year))
        
        # Build feature vector
        vector = []
        
        # Genre preferences (normalized)
        total_films = len(films)
        for genre in all_genres:
            vector.append(genre_counts.get(genre, 0) / total_films if total_films > 0 else 0)
        
        # Rating statistics (use defaults if no ratings)
        vector.append(np.mean(ratings) if len(ratings) > 0 else 3.0)
        vector.append(np.std(ratings) if len(ratings) > 1 else 0.5)
        
        # Temporal features
        vector.append(np.mean(years) / 2024.0 if years else 0.8)  # Normalize year
        vector.append(np.log1p(total_films))
        
        user_vectors[username] = np.array(vector)
    
    return user_vectors

def calculate_user_similarity(user_vectors: Dict) -> List[Dict]:
    """Calculate cosine similarity between users"""
    print("calculate_user_similarity called")
    if len(user_vectors) < 2:
        return []
    
    usernames = list(user_vectors.keys())
    vectors = np.array([user_vectors[u] for u in usernames])
    
    # Check if vectors are valid
    if vectors.shape[1] < 2:
        return []
    
    # Normalize vectors (handle zero variance)
    scaler = StandardScaler()
    try:
        vectors_normalized = scaler.fit_transform(vectors)
    except:
        # If standardization fails, use raw vectors
        vectors_normalized = vectors
    
    # Replace any NaN or Inf values
    vectors_normalized = np.nan_to_num(vectors_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors_normalized)
    
    # Create similarity pairs
    similarities = []
    for i in range(len(usernames)):
        for j in range(i + 1, len(usernames)):
            sim_score = float(similarity_matrix[i][j])
            # Clamp between 0 and 1 for percentage display
            sim_score = max(0.0, min(1.0, (sim_score + 1.0) / 2.0))  # Convert -1..1 to 0..1
            similarities.append({
                'user1': usernames[i],
                'user2': usernames[j],
                'similarity': sim_score
            })
    
    return sorted(similarities, key=lambda x: x['similarity'], reverse=True)

def find_taste_clusters(user_vectors: Dict) -> Dict:
    """Use KMeans to find taste clusters"""

    print("find_taste_clusters called")

    if len(user_vectors) < 3:
        return None
    
    usernames = list(user_vectors.keys())
    vectors = np.array([user_vectors[u] for u in usernames])
    
    # Cluster into 2-3 groups
    n_clusters = min(3, len(usernames))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(vectors)
    
    # Group users by cluster
    cluster_groups = {}
    for i, username in enumerate(usernames):
        cluster_id = int(clusters[i])
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(username)
    
    return cluster_groups

def build_rating_predictor(user_films: Dict[str, List[Film]]):
    """Build neural network to predict ratings"""

    print("build_rating_predictor called")


    X_train = []
    y_train = []
    
    for username, films in user_films.items():
        for film in films:
            if film.user_rating and film.genres:
                # Simple features
                features = [
                    len(film.genres),
                    1 if film.genres and 'Drama' in film.genres else 0,
                    1 if film.genres and 'Comedy' in film.genres else 0,
                    1 if film.genres and 'Action' in film.genres else 0,
                    1 if film.genres and 'Thriller' in film.genres else 0,
                    int(film.year) / 2024.0 if film.year and film.year.isdigit() else 0.5,
                ]
                X_train.append(features)
                y_train.append(1 if film.user_rating >= 4.0 else 0)
    
    if len(X_train) < 10:
        return None
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    # Build neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train silently
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    
    return model

def predict_genre_compatibility(model, genres=['Drama', 'Comedy', 'Action']) -> List[Dict]:
    """Predict group compatibility with different genres"""

    print("predict_genre_compatibility called")

    if model is None:
        return []
    
    predictions = []
    for genre in genres:
        features = [
            1,
            1 if genre == 'Drama' else 0,
            1 if genre == 'Comedy' else 0,
            1 if genre == 'Action' else 0,
            0,
            0.8
        ]
        features_array = np.array([features], dtype=np.float32)
        pred = model.predict(features_array, verbose=0)[0][0]
        predictions.append({
            'genre': genre,
            'compatibility_score': round(float(pred) * 100, 1)
        })
    
    return predictions

# ============= AGENT CLASSES =============

class ScraperAgent:
    """Independent agent that scrapes one user's films"""
    
    def __init__(self, username: str, max_pages: int = 3):
        self.username = username
        self.max_pages = max_pages
        self.films: List[Film] = []
        self.status = "idle"
        self.error = None
        self.thread_id = None
        
    def run(self) -> List[Film]:
        """Execute the scraping task"""
        self.status = "running"
        self.thread_id = threading.current_thread().name
        
        print(f"🤖 Agent [{self.username}] starting on thread {self.thread_id}")
        
        try:
            self.films = self._scrape_all_pages()
            self.status = "completed"
            print(f"✅ Agent [{self.username}] completed: {len(self.films)} films")
            return self.films
            
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            print(f"❌ Agent [{self.username}] failed: {e}")
            return []
    
    def _scrape_all_pages(self) -> List[Film]:
        """Scrape multiple pages for this user"""
        all_films = []
        
        for page in range(1, self.max_pages + 1):
            try:
                films = self._scrape_page(page)
                if not films:
                    break
                all_films.extend(films)
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️  Agent [{self.username}] error on page {page}: {e}")
                break
        
        return all_films
    
    def _scrape_page(self, page: int) -> List[Film]:
        """Scrape a single page"""
        url = f"{BASE_URL}/{self.username}/films/page/{page}/"
        
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0'
        }, timeout=10)
        
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        film_items = soup.select('li.griditem')
        
        films = []
        for item in film_items:
            try:
                film = self._parse_film_item(item)
                if film:
                    films.append(film)
            except Exception:
                continue
        
        return films
    
    def _parse_film_item(self, item) -> Film:
        """Parse a single film item"""
        react_div = item.select_one('div.react-component')
        if not react_div:
            return None
        
        slug = react_div.get('data-target-link', '')
        full_name = react_div.get('data-item-name', '')
        
        if not slug or not full_name:
            return None
        
        title, year = self._extract_title_and_year(full_name)
        return Film(slug=slug, title=title, year=year)
    
    @staticmethod
    def _extract_title_and_year(full_name: str) -> tuple:
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

class EnrichmentAgent:
    """Agent that enriches film data with additional metadata"""
    
    def __init__(self, films: List[Film], max_films: int = 30):
        self.films = films
        self.max_films = max_films
        
    def enrich(self) -> List[Film]:
        """Enrich films with genres, directors from detail pages"""
        print(f"🔍 Enrichment Agent: Processing {min(len(self.films), self.max_films)} films")
        
        enriched = []
        for film in self.films[:self.max_films]:
            try:
                details = self._get_film_details(film.slug)
                if details:
                    film.genres = details.get('genres', [])
                    film.director = details.get('director')
                enriched.append(film)
                time.sleep(0.5)
            except Exception as e:
                enriched.append(film)
                continue
        
        return enriched
    
    def _get_film_details(self, slug: str) -> Dict:
        """Get detailed film info"""
        url = f"{BASE_URL}{slug}"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        director_elem = soup.select_one('a[href*="/director/"]')
        director = director_elem.text.strip() if director_elem else None
        
        genres = []
        genre_links = soup.select('a[href*="/films/genre/"]')
        for genre_link in genre_links:
            genres.append(genre_link.text.strip())
        
        return {'director': director, 'genres': genres}

class AgentCoordinator:
    """Coordinates multiple scraper agents running in parallel"""
    
    def __init__(self, usernames: List[str], max_workers: int = 5):
        self.usernames = usernames
        self.max_workers = max_workers
        self.agents: List[ScraperAgent] = []
        self.results: Dict[str, List[Film]] = {}
        
    def execute(self) -> Dict[str, List[Film]]:
        """Execute all agents in parallel"""
        print(f"\n🎬 Coordinator: Starting {len(self.usernames)} agents in parallel")
        print(f"📊 Using {self.max_workers} worker threads")
        
        self.agents = [ScraperAgent(username) for username in self.usernames]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_agent = {
                executor.submit(agent.run): agent 
                for agent in self.agents
            }
            
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    films = future.result()
                    self.results[agent.username] = films
                except Exception as e:
                    print(f"❌ Agent [{agent.username}] exception: {e}")
                    self.results[agent.username] = []
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Coordinator: All agents completed in {elapsed:.2f}s")
        
        return self.results
    
    def get_status_report(self) -> Dict:
        """Get status of all agents"""
        return {
            'total_agents': len(self.agents),
            'completed': len([a for a in self.agents if a.status == 'completed']),
            'failed': len([a for a in self.agents if a.status == 'failed']),
            'agents': [
                {
                    'username': a.username,
                    'status': a.status,
                    'films_found': len(a.films),
                    'thread_id': a.thread_id,
                    'error': a.error
                }
                for a in self.agents
            ]
        }
    
class BasicAnalysisAgent:
    """Basic analysis"""

    def __init__(self, user_films: Dict[str, List[Film]]):
        self.user_films = user_films
        
    def analyze(self) -> Dict:
        """Perform basic analysis"""
        print("\n🔍 Analysis Agent: Starting basic analysis")
        
        # Basic analysis
        shared_films = self._find_shared_films()
        stats = self._calculate_statistics()
        top_years = self._find_top_years()
        
        print(f"✅ Analysis Agent: Complete")
        
        return {
            'user_count': len(self.user_films),
            'total_films': stats['total'],
            'shared_films': shared_films,
            'top_years': top_years,
            'stats_by_user': stats['by_user'],
        }
    
    def _find_shared_films(self) -> list[dict]:
        """Return films watched by all users (deduplicated and efficient)."""
        user_films = self.user_films
        if not user_films:
            return []

        iterator = iter(user_films.values())
        shared_slugs = {f.slug for f in next(iterator)}
        for films in iterator:
            shared_slugs.intersection_update(f.slug for f in films)
            if not shared_slugs:
                return []

        slug_to_film = {}
        for films in user_films.values():
            for f in films:
                if f.slug in shared_slugs and f.slug not in slug_to_film:
                    slug_to_film[f.slug] = f
                if len(slug_to_film) == len(shared_slugs):
                    break
            if len(slug_to_film) == len(shared_slugs):
                break

        return [
            {'title': f.title, 'year': f.year, 'slug': f.slug}
            for f in slug_to_film.values()
        ]
    
    def _calculate_statistics(self) -> Dict:
        """Calculate overall statistics"""
        total = sum(len(films) for films in self.user_films.values())
        
        by_user = {
            username: {
                'films': len(films),
                'has_years': len([f for f in films if f.year])
            }
            for username, films in self.user_films.items()
        }
        
        return {'total': total, 'by_user': by_user}
    
    def _find_top_years(self) -> List[Dict]:
        """Find most popular years"""
        all_years = [
            f.year 
            for films in self.user_films.values() 
            for f in films 
            if f.year
        ]
        
        year_counts = Counter(all_years)
        return [
            {'year': year, 'count': count}
            for year, count in year_counts.most_common(5)
        ]

class AnalysisAgent:
    """Agent that analyzes collected data with ML"""
    
    def __init__(self, user_films: Dict[str, List[Film]]):
        self._basic_agent = BasicAnalysisAgent(user_films)
        self.user_films = user_films
        
    def analyze(self) -> Dict:
        """Perform analysis with ML insights"""

        results = self._basic_agent.analyze()


        print("\n🔍 Analysis Agent: Starting analysis with ML")
        
        # ML Analysis
        ml_insights = {}
        
        # Only run ML if we have enriched data
        has_genres = any(
            any(f.genres for f in films) 
            for films in self.user_films.values()
        )

        if has_genres:
            print("🤖 Running ML analysis...")
            
            # User similarity
            user_vectors = create_user_vectors(self.user_films)
            if user_vectors:
                print(f"   Created vectors for {len(user_vectors)} users")
                # Debug: print vector dimensions
                for username, vector in list(user_vectors.items())[:2]:
                    print(f"   {username} vector shape: {vector.shape}, sample: {vector[:3]}")
                
                similarities = calculate_user_similarity(user_vectors)
                if similarities:
                    ml_insights['user_similarities'] = similarities[:3]
                    print(f"   Found {len(similarities)} similarity pairs")
            else:
                print("   ⚠️  Could not create user vectors (insufficient data)")
            
            # Taste clusters
            if user_vectors and len(user_vectors) >= 3:
                clusters = find_taste_clusters(user_vectors)
                ml_insights['taste_clusters'] = clusters
                print(f"   Created {len(clusters) if clusters else 0} taste clusters")
            
            # Neural network predictions
            model = build_rating_predictor(self.user_films)
            if model:
                genre_predictions = predict_genre_compatibility(model)
                ml_insights['genre_predictions'] = genre_predictions
                ml_insights['prediction_available'] = True
                print(f"   Trained neural network with predictions")
            else:
                ml_insights['prediction_available'] = False
                print(f"   ⚠️  Not enough rating data for neural network")
        else:
            print("⚠️  Skipping ML analysis - no genre data available")
            print("   💡 Use 'enrich: true' to enable ML features")
        
        print(f" ✅ Analysis Agent: Complete")
        
        results['ml_insights'] = ml_insights
        return results
    

# ============= API ENDPOINTS =============

@app.route('/api/analyze', methods=['POST'])
def analyze_group():
    """Main endpoint using parallel agents with ML"""
    try:
        data = request.get_json()
        usernames = data.get('usernames', [])
        enrich = data.get('enrich', False)  # Optional: enrich with genres/directors
        
        if len(usernames) < 2:
            return jsonify({'error': 'At least 2 usernames required'}), 400
        
        print(f"\n{'='*60}")
        print(f"🎯 New Analysis Request: {usernames}")
        print(f"{'='*60}")
        
        # Step 1: Coordinate parallel scraping
        coordinator = AgentCoordinator(usernames, max_workers=len(usernames))
        user_films = coordinator.execute()
        
        # Check for failures
        failed_users = [u for u, films in user_films.items() if not films]
        if failed_users:
            return jsonify({
                'error': f'Could not scrape users: {", ".join(failed_users)}'
            }), 404
        
        # Step 2: Optional enrichment (adds genres/directors for ML)
        if enrich:
            print("\n🔍 Enriching film data...")
            for username, films in user_films.items():
                enricher = EnrichmentAgent(films, max_films=30)
                user_films[username] = enricher.enrich()
        
        # Step 3: Analyze results with ML
        analyzer = AnalysisAgent(user_films)
        results = analyzer.analyze()
        
        # Step 4: Add agent status report
        results['agent_status'] = coordinator.get_status_report()
        
        print(f"\n{'='*60}")
        print(f"✨ Analysis Complete!")
        print(f"{'='*60}\n")

        print(results)
        
        return jsonify(results)
        
    except Exception as e:
        print(f"\n❌ Error in analyze_group: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'architecture': 'parallel-agents',
        'max_concurrent_agents': 5,
        'ml_enabled': True,
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Letterboxd Parallel Agents Backend with ML")
    print("="*60)
    print("Architecture: Multi-threaded agent coordination")
    print("Features:")
    print("  - Independent scraper agents per user")
    print("  - Parallel execution with ThreadPoolExecutor")
    print("  - Optional film enrichment for ML features")
    print("  - TensorFlow neural network predictions")
    print("  - NumPy feature engineering")
    print("  - scikit-learn clustering & similarity")
    print("="*60)
    print(f"TensorFlow: {tf.__version__}")
    print(f"NumPy: {np.__version__}")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8000)