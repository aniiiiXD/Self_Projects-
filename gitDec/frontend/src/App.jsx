import { useState } from 'react'

function App() {
  const [user, setUser] = useState("")
  const [profile, setProfile] = useState(null)
  const [events, setEvents] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    if (!user.trim()) {
      setError('Please enter a username');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const profileResponse = await fetch(`http://localhost:3000/profile/${user}`);
      const eventsResponse = await fetch(`http://localhost:3000/profile/${user}/events`);
      
      if (!profileResponse.ok) {
        throw new Error('User not found');
      }

      const profileData = await profileResponse.json();
      const eventsData = await eventsResponse.json();
      
      setProfile(profileData.data);
      setEvents(eventsData.profile);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-8 roboto-mono  ">
          GitHub Profile Viewer
        </h1>
        
        <div className="flex gap-4 mb-8">
          <input
            type="text"
            placeholder="Enter GitHub Username"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-colors"
            value={user}
            onChange={(e) => setUser(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button 
            className={`px-6 py-2 rounded-lg font-medium text-white transition-colors
              ${loading 
                ? 'bg-green-400 cursor-not-allowed' 
                : 'bg-green-500 hover:bg-green-600'}`}
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Search'}
          </button>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-600">
            {error}
          </div>
        )}

        {profile && (
          <div className="bg-white rounded-xl shadow-sm p-6">
            <div className="md:flex gap-6 mb-8">
              <img 
                src={profile.avatar_url} 
                alt="Profile" 
                className="w-32 h-32 md:w-40 md:h-40 rounded-full mx-auto md:mx-0"
              />
              
              <div className="mt-4 md:mt-0 text-center md:text-left flex-1">
                <h2 className="text-2xl font-bold text-gray-900">
                  {profile.name || profile.login}
                </h2>
                <p className="text-gray-600 mt-2">{profile.bio}</p>
                
                <div className="flex justify-center md:justify-start gap-6 mt-4">
                  <div className="text-center">
                    <span className="block text-xl font-semibold text-gray-900">
                      {profile.followers}
                    </span>
                    <span className="text-sm text-gray-600">Followers</span>
                  </div>
                  <div className="text-center">
                    <span className="block text-xl font-semibold text-gray-900">
                      {profile.following}
                    </span>
                    <span className="text-sm text-gray-600">Following</span>
                  </div>
                  <div className="text-center">
                    <span className="block text-xl font-semibold text-gray-900">
                      {profile.public_repos}
                    </span>
                    <span className="text-sm text-gray-600">Repos</span>
                  </div>
                </div>
              </div>
            </div>

            {events && events.length > 0 && (
              <div className="border-t pt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Recent Activity
                </h3>
                <div className="space-y-3">
                  {events.slice(0, 5).map((event, index) => (
                    <div 
                      key={index} 
                      className="flex flex-col md:flex-row md:items-center gap-2 md:gap-4 p-3 bg-gray-50 rounded-lg"
                    >
                      <span className="inline-block px-2 py-1 text-xs font-medium bg-blue-500 text-white rounded">
                        {event.type.replace('Event', '')}
                      </span>
                      <span className="text-blue-600 font-medium flex-1">
                        {event.repo.name}
                      </span>
                      <span className="text-sm text-gray-600">
                        {new Date(event.created_at).toLocaleDateString()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {profile.location && (
              <div className="mt-4 text-gray-600">
                📍 {profile.location}
              </div>
            )}

            {profile.blog && (
              <a 
                href={profile.blog}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-2 text-blue-500 hover:text-blue-600 inline-block"
              >
                🔗 {profile.blog}
              </a>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App