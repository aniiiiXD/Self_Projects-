import React from 'react';
import { CheckCircle, Clock, User } from 'lucide-react';

const VerifyBar = () => {
  // Example verification requests
  const verificationRequests = [
    {
      id: 1,
      name: "Ani",
      company: "DholakPur Inc.",
      skill: "Bakaiti Coordinator",
      status: "pending",
      timeAgo: "2 minutes ago",
      duration: "2 years",
      verifier: "Anuj Jha "
    }
  ];

  return (
    <div id="verify-section" className="min-h-screen bg-black p-8">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-2xl font-light text-white/90 mb-8 border-b border-white/10 pb-4">
          Verification Requests
        </h2>
        
        <div className="space-y-6">
          {verificationRequests.map((request) => (
            <div 
              key={request.id}
              className="bg-white/5 backdrop-blur-sm rounded-lg p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-start justify-between">
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center">
                      <User className="w-5 h-5 text-white/80" />
                    </div>
                    <div>
                      <h3 className="text-white/90 font-medium">{request.name}</h3>
                      <p className="text-white/60 text-sm">{request.company}</p>
                    </div>
                  </div>
                  
                  <div className="pl-13 space-y-2">
                    <div className="flex items-center space-x-2">
                      <div className="h-px w-4 bg-white/20"></div>
                      <p className="text-white/80">
                        Requesting verification for <span className="text-blue-400">{request.skill}</span>
                      </p>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <div className="h-px w-4 bg-white/20"></div>
                      <p className="text-white/60 text-sm">
                        Experience duration: {request.duration}
                      </p>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <div className="h-px w-4 bg-white/20"></div>
                      <p className="text-white/60 text-sm">
                        Verifier: {request.verifier}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex flex-col items-end space-y-2">
                  <div className="flex items-center space-x-2">
                    <Clock className="w-4 h-4 text-white/40" />
                    <span className="text-white/40 text-sm">{request.timeAgo}</span>
                  </div>
                  
                  <div className={`flex items-center space-x-2 px-3 py-1 rounded-full 
                    ${request.status === 'pending' ? 'bg-yellow-500/10 text-yellow-300' : 
                      request.status === 'verified' ? 'bg-green-500/10 text-green-300' : 
                      'bg-red-500/10 text-red-300'}`}
                  >
                    <div className="w-2 h-2 rounded-full bg-current"></div>
                    <span className="text-sm capitalize">{request.status}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default VerifyBar;