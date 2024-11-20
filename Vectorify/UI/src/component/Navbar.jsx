import React from 'react';
import { UserCircle, Key, LineChart } from 'lucide-react';

const Navbar = () => {
  const handleScrollToVerify = () => {
    const element = document.getElementById('verify-section');
    if (element) {
      element.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start'
      });
    }
  };

  const navItems = [
    { label: 'Profile', icon: UserCircle, onClick: handleScrollToVerify },
    { label: 'Manage Keys', icon: Key },
    { label: 'Status', icon: LineChart }
  ];

  return (
    <nav className="fixed top-0 w-full bg-black backdrop-blur-md border-b border-gray-800 shadow-lg z-50">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex justify-between items-center h-20">
          {/* Logo Section */}
          <div className="flex items-center space-x-2">
            <div className="text-2xl text-white font-semibold tracking-wide">
              <span className="text-indigo-500">V</span>ectorify
            </div>
          </div>

          {/* Navigation Items */}
          <div className="flex items-center space-x-8">
            {navItems.map((item) => (
              <button
                key={item.label}
                onClick={item.onClick}
                className="group flex items-center space-x-2 py-2 
                  text-sm text-gray-400 hover:text-gray-100
                  transition duration-300 relative cursor-pointer"
              >
                <item.icon className="w-5 h-5 stroke-[1.5] transition-transform duration-200 ease-in-out group-hover:scale-110 group-hover:stroke-indigo-500" />
                <span className="tracking-wider font-light">{item.label}</span>
                <div className="absolute -bottom-0.5 left-0 w-0 h-px bg-indigo-500 
                  transition-all duration-300 group-hover:w-full"></div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
