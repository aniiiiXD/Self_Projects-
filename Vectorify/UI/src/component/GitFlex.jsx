import React from 'react';
import { Trophy, Users, GitCommitHorizontal, GitPullRequest } from 'lucide-react';

const GitFlex = () => {
  const globalGroup = [
    { id: 1, name: "MAdhusudhan", avatar: "/api/placeholder/50/50", rank: 1, commits: 250, contributions: 120 },
    { id: 2, name: "SR Ghorpade", avatar: "/api/placeholder/50/50", rank: 2, commits: 200, contributions: 100 },
    { id: 3, name: "Rekha ", avatar: "/api/placeholder/50/50", rank: 3, commits: 180, contributions: 90 },
    { id: 4, name: "MAyukh", avatar: "/api/placeholder/50/50", rank: 4, commits: 150, contributions: 80 },
    { id: 5, name: "Bata", avatar: "/api/placeholder/50/50", rank: 5, commits: 130, contributions: 70 },
  ];

  const localGroup = [
    { id: 1, name: "Tony ", avatar: "/api/placeholder/50/50", rank: 1, commits: 220, contributions: 110 },
    { id: 2, name: "Ronnie ", avatar: "/api/placeholder/50/50", rank: 2, commits: 190, contributions: 95 },
    { id: 3, name: "Shantanu dey ", avatar: "/api/placeholder/50/50", rank: 3, commits: 170, contributions: 85 },
    { id: 4, name: "Ravi Raghu", avatar: "/api/placeholder/50/50", rank: 4, commits: 160, contributions: 80 },
    { id: 5, name: "Kaushik Saha", avatar: "/api/placeholder/50/50", rank: 5, commits: 140, contributions: 75 },
  ];

  const LeaderboardSection = ({ title, users, icon: Icon }) => (
    <div className="space-y-6">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center">
          <Icon className="w-5 h-5 text-white/80" />
        </div>
        <h2 className="text-2xl font-light text-white/90">{title}</h2>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {users.map((user) => (
          <div
            key={user.id}
            className="bg-white/5 backdrop-blur-sm rounded-lg p-6 hover:bg-white/10 transition-all duration-300"
          >
            <div className="flex items-start space-x-4">
              <div className="relative">
                <img
                  src={user.avatar}
                  alt={user.name}
                  className="w-12 h-12 rounded-full bg-gradient-to-br from-gray-700 to-gray-800"
                />
                <div className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-blue-500/20 flex items-center justify-center">
                  <span className="text-sm text-blue-300">#{user.rank}</span>
                </div>
              </div>
              
              <div className="flex-1 space-y-2">
                <h3 className="text-white/90 font-medium">{user.name}</h3>
                
                <div className="space-y-2">
                  <div className="flex items-center space-x-2">
                    <div className="h-px w-4 bg-white/20"></div>
                    <div className="flex items-center space-x-2">
                      <GitCommitHorizontal className="w-4 h-4 text-green-400" />
                      <p className="text-white/60 text-sm">
                        {user.commits} commits
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <div className="h-px w-4 bg-white/20"></div>
                    <div className="flex items-center space-x-2">
                      <GitPullRequest className="w-4 h-4 text-blue-400" />
                      <p className="text-white/60 text-sm">
                        {user.contributions} contributions
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-black p-8">
      <div className="max-w-6xl mx-auto space-y-12">
        <div className="text-center space-y-2">
          <div className="w-16 h-16 mx-auto rounded-full bg-gradient-to-br from-blue-600/20 to-purple-600/20 flex items-center justify-center mb-4">
            <Trophy className="w-8 h-8 text-blue-400" />
          </div>
          <h1 className="text-3xl font-light text-white/90">
            GitHub Flex Leaderboard
          </h1>
          <p className="text-white/60">Track and celebrate top contributors</p>
        </div>

        <LeaderboardSection 
          title="Global Rankings" 
          users={globalGroup}
          icon={Users}
        />
        
        <LeaderboardSection 
          title="Local Rankings" 
          users={localGroup}
          icon={Users}
        />
      </div>
    </div>
  );
};

export default GitFlex;