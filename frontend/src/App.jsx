import React, { useState } from 'react';
import { Users, Film, Star, TrendingUp, Award, X, Plus, Search, AlertCircle, Loader } from 'lucide-react';

export default function LetterboxdGroupStats() {
  const [usernames, setUsernames] = useState([]);
  const [currentInput, setCurrentInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');

  // Change this to your Flask backend URL when deployed
  const API_URL = 'http://127.0.0.1:5000/';

  const addUsername = () => {
    if (currentInput.trim() && !usernames.includes(currentInput.trim())) {
      setUsernames([...usernames, currentInput.trim()]);
      setCurrentInput('');
      setError('');
    }
  };

  const removeUsername = (index) => {
    setUsernames(usernames.filter((_, i) => i !== index));
    setStats(null);
  };

  const analyzeGroup = async () => {
    if (usernames.length < 2) {
      setError('Please add at least 2 usernames');
      return;
    }

    setLoading(true);
    setError('');
    setStats(null);

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ usernames }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch data from server');
      }

      const data = await response.json();
      setStats(data);
    } catch (err) {
      setError(err.message || 'Failed to analyze. Please check usernames and try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-indigo-900 to-blue-900 p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Film className="w-12 h-12 text-orange-400" />
            <h1 className="text-4xl md:text-5xl font-bold text-white">
              Letterboxd Group Stats
            </h1>
          </div>
          <p className="text-indigo-200 text-lg">
            Analyze shared movie tastes across multiple Letterboxd users
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 md:p-8 mb-8 shadow-2xl border border-white/20">
          <h2 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
            <Users className="w-6 h-6" />
            Add Usernames
          </h2>
          
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={currentInput}
              onChange={(e) => setCurrentInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && addUsername()}
              placeholder="Enter Letterboxd username"
              className="flex-1 px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white placeholder-indigo-200 focus:outline-none focus:ring-2 focus:ring-orange-400"
            />
            <button
              onClick={addUsername}
              className="px-6 py-3 bg-orange-500 hover:bg-orange-600 text-white rounded-lg font-semibold transition flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              Add
            </button>
          </div>

          <div className="flex flex-wrap gap-2 mb-4 min-h-[40px]">
            {usernames.map((username, index) => (
              <div
                key={index}
                className="bg-indigo-600 text-white px-4 py-2 rounded-full flex items-center gap-2 animate-in fade-in"
              >
                <span>@{username}</span>
                <button
                  onClick={() => removeUsername(index)}
                  className="hover:bg-indigo-700 rounded-full p-1 transition"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>

          <button
            onClick={analyzeGroup}
            disabled={loading || usernames.length < 2}
            className="w-full px-6 py-4 bg-gradient-to-r from-orange-500 to-pink-500 hover:from-orange-600 hover:to-pink-600 disabled:from-gray-500 disabled:to-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-bold text-lg transition flex items-center justify-center gap-2 shadow-lg"
          >
            {loading ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Analyze Group
              </>
            )}
          </button>

          {error && (
            <div className="mt-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-200 flex items-start gap-2">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Results Section */}
        {stats && (
          <div className="space-y-6 animate-in fade-in">
            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <StatCard
                icon={<Users className="w-6 h-6" />}
                label="Users Analyzed"
                value={stats.user_count}
                color="bg-blue-500"
              />
              <StatCard
                icon={<Film className="w-6 h-6" />}
                label="Total Films"
                value={stats.total_films}
                color="bg-purple-500"
              />
              <StatCard
                icon={<Star className="w-6 h-6" />}
                label="Avg Rating"
                value={stats.avg_rating}
                color="bg-orange-500"
              />
              <StatCard
                icon={<TrendingUp className="w-6 h-6" />}
                label="Shared Films"
                value={stats.shared_films?.length || 0}
                color="bg-pink-500"
              />
            </div>

            {/* Shared Films */}
            {stats.shared_films && stats.shared_films.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-white/20">
                <h3 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
                  <Film className="w-6 h-6 text-orange-400" />
                  Films Watched by Everyone ({stats.shared_films.length})
                </h3>
                <div className="space-y-3">
                  {stats.shared_films.slice(0, 10).map((film, i) => (
                    <div key={i} className="bg-white/5 rounded-lg p-4 hover:bg-white/10 transition">
                      <div className="flex justify-between items-start">
                        <div>
                          <span className="text-white font-medium text-lg">{film.title}</span>
                          {film.year && <span className="text-indigo-300 ml-2">({film.year})</span>}
                        </div>
                        {film.avg_rating && (
                          <div className="flex items-center gap-1">
                            <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                            <span className="text-white">{film.avg_rating}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top Rated Films */}
            {stats.top_rated && stats.top_rated.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-white/20">
                <h3 className="text-2xl font-semibold text-white mb-4 flex items-center gap-2">
                  <Award className="w-6 h-6 text-orange-400" />
                  Top Rated Films
                </h3>
                <div className="space-y-3">
                  {stats.top_rated.map((film, i) => (
                    <div key={i} className="bg-white/5 rounded-lg p-4 hover:bg-white/10 transition">
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-3">
                          <span className="text-orange-400 font-bold text-xl">#{i + 1}</span>
                          <div>
                            <span className="text-white font-medium">{film.title}</span>
                            {film.year && <span className="text-indigo-300 ml-2">({film.year})</span>}
                          </div>
                        </div>
                        <div className="flex items-center gap-1">
                          <Star className="w-5 h-5 text-yellow-400 fill-yellow-400" />
                          <span className="text-white font-semibold">{film.rating}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Favorite Directors */}
            {stats.favorite_directors && stats.favorite_directors.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-white/20">
                <h3 className="text-2xl font-semibold text-white mb-4">Most Watched Directors</h3>
                <div className="space-y-3">
                  {stats.favorite_directors.map((dir, i) => (
                    <div key={i} className="bg-white/5 rounded-lg p-4">
                      <div className="flex justify-between items-center">
                        <span className="text-white font-medium">{dir.name}</span>
                        <span className="text-indigo-300">{dir.count} films</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Genre Breakdown */}
            {stats.genre_breakdown && stats.genre_breakdown.length > 0 && (
              <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-white/20">
                <h3 className="text-2xl font-semibold text-white mb-4">Genre Distribution</h3>
                <div className="space-y-4">
                  {stats.genre_breakdown.map((genre, i) => (
                    <div key={i}>
                      <div className="flex justify-between text-white mb-2">
                        <span>{genre.genre}</span>
                        <span>{genre.percentage}%</span>
                      </div>
                      <div className="w-full bg-white/20 rounded-full h-3">
                        <div
                          className="bg-gradient-to-r from-orange-400 to-pink-500 h-3 rounded-full transition-all duration-500"
                          style={{ width: `${genre.percentage}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ML INSIGHTS SECTION */}
            {stats.ml_insights && (
              <>
                {/* User Similarity */}
                {stats.ml_insights.user_similarities && stats.ml_insights.user_similarities.length > 0 && (
                  <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-purple-400/30">
                    <h3 className="text-2xl font-semibold text-white mb-2 flex items-center gap-2">
                      🤖 AI-Powered User Similarity
                    </h3>
                    <p className="text-purple-200 text-sm mb-4">Using cosine similarity on viewing patterns</p>
                    <div className="space-y-3">
                      {stats.ml_insights.user_similarities.map((sim, i) => (
                        <div key={i} className="bg-white/5 rounded-lg p-4">
                          <div className="flex justify-between items-center mb-2">
                            <div className="text-white">
                              <span className="font-semibold">@{sim.user1}</span>
                              <span className="mx-2">↔️</span>
                              <span className="font-semibold">@{sim.user2}</span>
                            </div>
                            <span className="text-green-400 font-bold">
                              {(sim.similarity * 100).toFixed(1)}% match
                            </span>
                          </div>
                          <div className="w-full bg-white/20 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-green-400 to-emerald-500 h-2 rounded-full"
                              style={{ width: `${sim.similarity * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Taste Clusters */}
                {stats.ml_insights.taste_clusters && (
                  <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-blue-400/30">
                    <h3 className="text-2xl font-semibold text-white mb-2 flex items-center gap-2">
                      🎯 Taste Clusters (K-Means)
                    </h3>
                    <p className="text-blue-200 text-sm mb-4">Machine learning grouped users by similar tastes</p>
                    <div className="space-y-3">
                      {Object.entries(stats.ml_insights.taste_clusters).map(([cluster, users], i) => (
                        <div key={i} className="bg-white/5 rounded-lg p-4">
                          <div className="text-cyan-400 font-semibold mb-2">
                            Cluster {parseInt(cluster) + 1}
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {users.map((user, j) => (
                              <span key={j} className="bg-cyan-600 text-white px-3 py-1 rounded-full text-sm">
                                @{user}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Neural Network Predictions */}
                {stats.ml_insights.genre_predictions && stats.ml_insights.genre_predictions.length > 0 && (
                  <div className="bg-gradient-to-br from-orange-500/20 to-red-500/20 backdrop-blur-lg rounded-2xl p-6 shadow-2xl border border-orange-400/30">
                    <h3 className="text-2xl font-semibold text-white mb-2 flex items-center gap-2">
                      🧠 Neural Network Predictions
                    </h3>
                    <p className="text-orange-200 text-sm mb-4">TensorFlow predicts your group's compatibility with genres</p>
                    <div className="space-y-4">
                      {stats.ml_insights.genre_predictions.map((pred, i) => (
                        <div key={i}>
                          <div className="flex justify-between text-white mb-2">
                            <span className="font-medium">{pred.genre} Films</span>
                            <span className={`font-bold ${
                              pred.compatibility_score >= 70 ? 'text-green-400' :
                              pred.compatibility_score >= 50 ? 'text-yellow-400' :
                              'text-red-400'
                            }`}>
                              {pred.compatibility_score}% likely to enjoy
                            </span>
                          </div>
                          <div className="w-full bg-white/20 rounded-full h-3">
                            <div
                              className={`h-3 rounded-full transition-all duration-500 ${
                                pred.compatibility_score >= 70 ? 'bg-gradient-to-r from-green-400 to-emerald-500' :
                                pred.compatibility_score >= 50 ? 'bg-gradient-to-r from-yellow-400 to-orange-500' :
                                'bg-gradient-to-r from-red-400 to-pink-500'
                              }`}
                              style={{ width: `${pred.compatibility_score}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon, label, value, color }) {
  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 shadow-xl border border-white/20">
      <div className={`${color} w-12 h-12 rounded-lg flex items-center justify-center mb-3 text-white`}>
        {icon}
      </div>
      <div className="text-indigo-200 text-sm mb-1">{label}</div>
      <div className="text-white text-3xl font-bold">{value}</div>
    </div>
  );
}