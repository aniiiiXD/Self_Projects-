import React, { useState } from 'react';

const Login = () => {

  const [userName , setUser] = useState("")
  const [pass , setPass] = useState("")

  

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 min-h-screen flex items-center justify-center">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden flex w-full max-w-5xl">
        
        <div className="w-1/2 hidden lg:block">
          <img
            src="https://www.bankofbaroda.in/-/media/project/bob/countrywebsites/india/blogs/images/22-05/medical-ai.jpg"
            alt="Medical AI"
            className="object-cover w-full h-full"
          />
        </div>

        
        <div className="lg:w-1/2 w-full p-10 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col justify-center">
          <h1 className="text-4xl font-bold mb-8 text-center">Welcome Back</h1>
          <form className="space-y-6">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
                Username
              </label>
              <input
                type="text"
                id="username"
                name="username"
                className="w-full px-4 py-2 bg-gray-800 text-white rounded-md border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter your username"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Password
              </label>
              <input
                type="password"
                id="password"
                name="password"
                className="w-full px-4 py-2 bg-gray-800 text-white rounded-md border border-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter your password"
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="remember"
                  name="remember"
                  className="text-red-500 focus:ring-red-500 border-gray-600 rounded"
                />
                <label htmlFor="remember" className="text-gray-400 ml-2">
                  Remember Me
                </label>
              </div>
              <a href="#" className="text-sm text-blue-400 hover:underline">
                Forgot Password?
              </a>
            </div>

            <button
              type="submit"
              className="w-full py-3 bg-red-600 hover:bg-red-700 text-white rounded-md font-semibold transition duration-200"
            >
              Login
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className="text-gray-400">
              Don't have an account?{' '}
              <a href="#" className="text-blue-400 hover:underline">
                Sign up here
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
