import React from 'react';
import { Shield, Brain, Lock, Clock } from 'lucide-react';
import './utils.css'

const Hero = () => {
  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 min-h-screen">

      {/* Navbar */}
      <nav className="bg-transparent py-6">
        <div className="container mx-auto flex justify-between items-center px-6">
          <a href="#" className="text-3xl font-bold text-white">
            HealthAI
          </a>
          <div className="flex space-x-6">
            <a href="#features" className="text-lg text-slate-300 hover:text-white transition duration-300">
              Features
            </a>
            <a href="#about" className="text-lg text-slate-300 hover:text-white transition duration-300">
              About
            </a>
            <a href="#contact" className="text-lg text-slate-300 hover:text-white transition duration-300">
              Contact
            </a>
            <a href="/login" className=''>
              <button className="bg-gradient-to-r from-indigo-500 to-purple-500 px-6 py-2 rounded-lg shadow-lg hover:shadow-indigo-500/50 hover:-translate-y-1 transition-all duration-300">
                Sign Up
              </button>
            </a>

          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500/20 rounded-full blur-3xl"></div>
        <div className="absolute top-40 -left-40 w-80 h-80 bg-indigo-500/20 rounded-full blur-3xl"></div>
      </div>

      <div className="container mx-auto px-6 md:px-12 py-20 relative">
        <div className="md:flex items-center justify-between">
          <div className="md:w-1/2 z-10 text-white space-y-6">
            <h1 className="playfair-display text-5xl font-bold leading-tight mb-4">
              Next-Gen Healthcare Intelligence
            </h1>
            <p className="text-lg leading-relaxed">
              Empowering healthcare professionals with cutting-edge AI technology
              for superior patient care and diagnostic precision.
            </p>
            <div className="flex space-x-4 mt-6">
              <button className="bg-gradient-to-r from-indigo-500 to-purple-500 px-8 py-3 rounded-lg shadow-lg hover:shadow-indigo-500/50 hover:-translate-y-1 transition-all duration-300">
                Get Started
              </button>
              <button className="backdrop-blur-sm bg-white/10 px-8 py-3 rounded-lg hover:bg-white/20 transition-all duration-300 text-white">
                Learn More
              </button>
            </div>
          </div>

          <div className="md:w-1/2 grid grid-cols-2 gap-8 mt-12 md:mt-0">
            {/* Glass cards */}
            <div className="backdrop-blur-sm bg-white/10 p-6 rounded-2xl hover:bg-white/[0.2] transition-all duration-300 border border-white/10 shadow-lg">
              <Brain className="w-10 h-10 text-indigo-400 mb-4" />
              <h3 className="text-white font-semibold text-xl mb-2">AI Analysis</h3>
              <p className="text-slate-300">Real-time diagnostic insights powered by advanced ML</p>
            </div>
            <div className="backdrop-blur-sm bg-white/10 p-6 rounded-2xl hover:bg-white/[0.2] transition-all duration-300 border border-white/10 shadow-lg">
              <Lock className="w-10 h-10 text-indigo-400 mb-4" />
              <h3 className="text-white font-semibold text-xl mb-2">Secure Platform</h3>
              <p className="text-slate-300">Enterprise-grade security with HIPAA compliance</p>
            </div>
            <div className="backdrop-blur-sm bg-white/10 p-6 rounded-2xl hover:bg-white/[0.2] transition-all duration-300 border border-white/10 shadow-lg">
              <Clock className="w-10 h-10 text-indigo-400 mb-4" />
              <h3 className="text-white font-semibold text-xl mb-2">Real-time Results</h3>
              <p className="text-slate-300">Instant analysis with sub-second response time</p>
            </div>
            <div className="backdrop-blur-sm bg-white/10 p-6 rounded-2xl hover:bg-white/[0.2] transition-all duration-300 border border-white/10 shadow-lg">
              <Shield className="w-10 h-10 text-indigo-400 mb-4" />
              <h3 className="text-white font-semibold text-xl mb-2">Privacy First</h3>
              <p className="text-slate-300">End-to-end encryption for all patient data</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
