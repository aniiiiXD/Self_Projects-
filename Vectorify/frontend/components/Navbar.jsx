import React from 'react';

const Navbar = () => {
  const navItems = [
    { label: 'Profile', icon: '👤' },
    { label: 'Manage Keys', icon: '🔑' },
    { label: 'Status', icon: '📊' }
  ];

  return (
    <nav className="bg-gray-900 text-gray-300 shadow-lg">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          {/* Logo Section */}
          <div className="flex items-center">
            <h1 className="text-2xl font-bold tracking-wider">
              <span className="text-white">VEC</span>
              <span className="text-gray-400">tor</span>
            </h1>
          </div>

          {/* Navigation Items */}
          <div className="flex items-center space-x-4">
            {navItems.map((item) => (
              <button
                key={item.label}
                className="flex items-center px-4 py-2 rounded-lg
                         text-sm font-medium
                         transition-all duration-200
                         hover:bg-gray-800 hover:text-white
                         focus:outline-none focus:ring-2 focus:ring-gray-500
                         active:bg-gray-700"
              >
                <span className="mr-2">{item.icon}</span>
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
