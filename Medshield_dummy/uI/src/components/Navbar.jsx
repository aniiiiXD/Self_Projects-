import React from 'react';

function Navbar() {
  return (
    <div className="min-h-screen bg-black text-white flex flex-col justify-between">
      {/* Navigation Bar */}
      <nav className="p-6">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="text-xl font-bold">
            <span className="text-3xl playwrite-gb-s">Med</span>
            <span className="text-5xl"> AI</span>
          </div>
          <div className="flex gap-8">
            <a href="#" className="hover:text-gray-300 transition-colors">About</a>
            <a href="#" className="hover:text-gray-300 transition-colors">Team</a>
            <a href="#" className="hover:text-gray-300 transition-colors">Careers</a>
          </div>
        </div>
      </nav>

      
    
      
      <footer className="flex gap-6 p-6 justify-center mb-36 text-center">

        <a href="#" className="text-2xl  underline">
          Learn More about Us!
        </a>
        <a href="#" className="text-2xl  underline flex items-center gap-2">
          Let's Connect Together
          <svg className="w-4 h-4 transform rotate-45" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
        </a>
      </footer>
    </div>
  );
}

export default Navbar;
