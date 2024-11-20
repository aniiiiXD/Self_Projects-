import React, { useState } from 'react';
import { Key, Lock, Eye, EyeOff } from 'lucide-react';

const KeysBar = () => {
  // Sample cryptographic keys data
  const keys = [
    {
      id: 1,
      wallet: "Metamask",
      keyName: "Metamask Main Key",
      status: "connected",
      lastAccessed: "1 hour ago"
    },
    {
      id: 2,
      wallet: "Phantom",
      keyName: "Phantom Secure Key",
      status: "disconnected",
      lastAccessed: "2 days ago"
    }
  ];

  const [showPasswordPrompt, setShowPasswordPrompt] = useState(false);
  const [activeKeyId, setActiveKeyId] = useState(null);

  const handleViewKey = (keyId) => {
    setShowPasswordPrompt(true);
    setActiveKeyId(keyId);
  };

  const handleClosePrompt = () => {
    setShowPasswordPrompt(false);
    setActiveKeyId(null);
  };

  return (
    <div className="min-h-screen bg-black p-8">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-2xl font-light text-white/90 mb-8 border-b border-white/10 pb-4">
          Manage Cryptographic Keys
        </h2>

        <div className="space-y-6">
          {keys.map((key) => (
            <div 
              key={key.id}
              className="bg-white/5 backdrop-blur-sm rounded-lg p-6 hover:bg-white/10 transition-all duration-300"
            >
              <div className="flex items-start justify-between">
                <div className="space-y-2">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center">
                      <Key className="w-5 h-5 text-white/80" />
                    </div>
                    <div>
                      <h3 className="text-white/90 font-medium">{key.wallet}</h3>
                      <p className="text-white/60 text-sm">{key.keyName}</p>
                    </div>
                  </div>

                  <div className="pl-13 space-y-2">
                    <div className="flex items-center space-x-2">
                      <div className="h-px w-4 bg-white/20"></div>
                      <p className="text-white/80">
                        Status: <span className={`${key.status === 'connected' ? 'text-green-400' : 'text-red-400'}`}>{key.status}</span>
                      </p>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="h-px w-4 bg-white/20"></div>
                      <p className="text-white/60 text-sm">
                        Last accessed: {key.lastAccessed}
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <button
                    onClick={() => handleViewKey(key.id)}
                    className="flex items-center space-x-2 px-4 py-2 rounded-md bg-blue-600/10 text-blue-300 hover:bg-blue-600/20 transition duration-300"
                  >
                    <Lock className="w-4 h-4" />
                    <span className="text-sm">View Key</span>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {showPasswordPrompt && (
          <div className="fixed inset-0 flex items-center justify-center bg-black/50">
            <div className="bg-black p-6 rounded-lg max-w-sm w-full space-y-4">
              <h3 className="text-lg font-medium text-white/90">Enter Password</h3>
              <p className="text-sm text-white/60">Please enter your password to view the key details.</p>
              <input
                type="password"
                className="w-full px-4 py-2 rounded-md bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter password"
              />
              <div className="flex items-center justify-end space-x-2">
                <button
                  onClick={handleClosePrompt}
                  className="px-4 py-2 text-sm rounded-md bg-red-500/10 text-red-400 hover:bg-red-500/20 transition duration-300"
                >
                  Cancel
                </button>
                <button
                  onClick={() => alert(`Password accepted. Showing key for ID: ${activeKeyId}`)}
                  className="px-4 py-2 text-sm rounded-md bg-blue-600/10 text-blue-300 hover:bg-blue-600/20 transition duration-300"
                >
                  View Key
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default KeysBar;
    